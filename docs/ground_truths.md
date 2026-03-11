# Ground Truths

Key findings and verified facts for the ROSE project. This file is updated
as development progresses so that future decisions build on established knowledge.

## Architecture

- **Analyzer planner location**: `/Users/m2d/git/analyzer/analyzer_tools/planner/`
  - Core files: `experiment_design.py`, `optimizer.py`, `instrument.py`, `mcmc_sampler.py`, `report.py`, `cli.py`
  - Algorithm: Bayesian experimental design following Treece et al., J. Appl. Cryst. (2019), 52, 47–59
  - Two entropy methods: MVN (multivariate normal) and KDN (kernel density estimation)
  - Uses bumps DREAM sampler for MCMC posterior sampling
  - **bumps `parallel` support**: `perform_mcmc()` accepts a `parallel` parameter
    forwarded to `bumps.fitters.fit()`. `0` = all CPUs, `1` = single CPU.
    Sequential optimization uses `parallel=0` (all CPUs per fit); parallel
    optimization uses `parallel=1` (avoids oversubscription with outer
    `ProcessPoolExecutor`).
  - **Combined parallelization**: `optimize_parallel()` submits every
    `(value, realization)` pair as an independent task via
    `_evaluate_single_realization()`. Total tasks = `len(param_values) ×
    realizations` (e.g. 7 values × 5 realizations = 35 tasks), keeping
    all workers busy even when there are few hypothesis values.
    `--workers N` CLI flag overrides the default auto-scaling
    (`min(total_tasks, cpu_count, MAX_WORKERS)`).

- **AuRE web app location**: `/Users/m2d/git/aure/`
  - Flask 3.0+ with Blueprint-based routing
  - LangGraph pipeline: intake → analysis → modeling → fitting → evaluation → refinement
  - FastMCP server for AI tool integration
  - Plugin integration options: Blueprint mount at `/rose`, MCP tools, or LangGraph nodes

## Data Formats

- **Model files**: YAML (`.yaml`/`.yml`) or JSON (`.json`) declarative descriptions.
  No code execution — models are parsed with `yaml.safe_load` / `json.loads`.
  Schema: `name` (optional), `layers` (list of layer dicts), `experiment` (optional settings).
  Each layer has `name`, `rho`, optional `irho`, `thickness`, `interface`, and `fit` map.
  Fit ranges: `fit: {param: [min, max]}` for `thickness`, `interface`, `rho`, `irho`.
- **Data files**: 4-column text: Q, Reflectivity, dReflectivity, dQ
- **Results**: JSON with `[param_value, info_gain, std_info_gain]` arrays plus per-realization SLD/reflectivity data

## Information Gain Algorithm

```
H_prior = Σ log₂(p_max - p_min)  for all parameters of interest

For each parameter value to test:
    For each noise realization:
        1. Draw "true" fitted-parameter values uniformly from prior bounds
        2. Simulate reflectivity from the drawn truth
        3. Add noise to get synthetic measurement
        4. Restore YAML initial values as MCMC starting point
        5. Run MCMC (DREAM) on the noisy data
        6. Extract marginal posterior samples for parameters of interest
        7. Compute H_posterior via MVN or KDE
        8. ΔH = H_prior - H_posterior

    Average ΔH over realizations → information gain for this parameter value

Output: parameter value with maximum average ΔH
```

**Key design choice**: The "true" values for each realization are drawn
from the prior (step 1), not taken from the YAML initial values. This
marginalizes the information gain over the unknown truth, making the
result independent of the initial `rho`, `thickness`, etc. values in
the model file. The initial values in the YAML are only used as MCMC
starting points (step 4). Bug fix applied 2026-03-11; previously the
initial values were used as truth, which caused the ΔH to depend on
the arbitrary choice of initial parameter values.

## Decisions

- **Planner integration**: Copy and adapt from analyzer (not a dependency)
- **Use-case 2**: LLM-based (LangChain) text → refl1d model generation
- **Web framework**: Flask with Blueprint pattern (same as AuRE)
- **CLI framework**: Click

## Phase 1 — Planner Port

### Bug found and fixed
- `ExperimentDesigner.__init__` called `_get_parameters()` before `_model_parameters_to_dict()`, but `_get_parameters()` references `self.all_model_parameters` for the POI check. Fixed by swapping the init order: `all_model_parameters` first, then `parameters`.

### Key porting decisions
- **`_get_sld_contour`** was inlined into `optimizer.py` (originally from `analyzer_tools.utils.model_utils`) to avoid the analyzer dependency. Uses `refl1d.uncertainty.calc_errors`, `align_profiles`, `_build_profile_matrix`.
- **Pydantic removed**: Analyzer used `BaseModel` for `SampleParameter` and `ExperimentRealization`. ROSE uses plain dicts + dataclasses in `core/types.py`.
- **bumps is transitive**: Not listed in direct deps; comes via refl1d >=1.0.0.

### CLI structure
- `rose` — Click group (top-level)
- `rose inspect <MODEL_FILE>` — loads model, shows variable/fixed params
- `rose optimize <MODEL_FILE>` — runs full optimization (all settings from YAML)
- `rose report --result-file ... --output-dir ...` — regenerates plots from JSON

### YAML-driven configuration
- All experiment and optimization settings moved from CLI flags to the YAML model file.
- **Three-section YAML schema**: `layers`, `experiment`, `optimization`.
- **`experiment` section** (all optional with defaults): `q_min` (0.008), `q_max` (0.2), `q_points` (50), `dq_over_q` (0.025), `relative_error` (0.10), `step_interfaces` (None).
- **`optimization` section**: `param` (required), `param_values` (required), `parameters_of_interest` (optional), `num_realizations` (3), `mcmc_steps` (2000), `entropy_method` ("kdn").
- **Remaining CLI options**: `MODEL_FILE` (positional), `--data-file`, `--output-dir`, `--parallel/--sequential`, `--workers N` (max parallel processes, default auto), `--verbose`.
- **Validation**: `_validate_experiment()` and `_validate_optimization()` in model_loader.py enforce bounds regardless of entry point (CLI, API, or web).
- **Constants**: `EXPERIMENT_DEFAULTS` and `OPTIMIZATION_DEFAULTS` dicts in model_loader.py.
- **Rationale**: Cleaner CLI, self-contained model files, consistent validation across interfaces.

### Test coverage (Phase 1 end)
- 83 tests, all passing (including 9 security tests)
- Modules tested: experiment_design (83%), instrument (92%), model_loader (94%), report (97%), cli (36%), core/types (100%), core/config (94%)
- `mcmc_sampler` and `optimizer` at 0% — these require MCMC which is slow; tested indirectly via CLI integration

### Prior sampling bug fix (2026-03-11)
- **Bug**: `evaluate_param()` used YAML initial values as "ground truth" for simulating synthetic data. Changing an initial value (e.g. material rho from 5 to 4) changed the information gain, even though the fit range covered both values.
- **Fix**: Each noise realization now draws "true" fitted-parameter values uniformly from the prior bounds via `draw_truth_from_prior(rng)`. This marginalizes the information gain over all possible truths.
- **RNG safety**: Explicit `np.random.Generator` is threaded through `draw_truth_from_prior()` and `add_noise()`. In `evaluate_param()`, the RNG is seeded from the parameter value, giving each parallel worker a unique but reproducible stream.
- **State restoration**: A `finally` block in the realization loop calls `restore_parameter_values()` so the model is always left in a clean state.
- **Failed realizations**: Now recorded as `NaN` (not `0.0`) and excluded via `np.nanmean`/`np.nanstd` to avoid biasing the mean downward.

### YAML/JSON model migration
- Replaced `importlib`-based Python model loading with declarative YAML/JSON parsing.
- **Eliminated CRITICAL security risk**: arbitrary code execution via `importlib.exec_module` is no longer possible.
- New functions: `load_model_description()`, `build_experiment()`, `_validate_layers()` — all in `model_loader.py`.
- `build_experiment()` constructs refl1d layer stacks programmatically from the dict: `SLD(name, rho) → material(thickness, interface) → sample via | operator → Experiment(sample, probe)`.
- `inspect_model()` and `load_experiment()` keep the same signatures but now delegate to the YAML/JSON pipeline.
- Example models converted: `layer_a_on_b.yaml`, `cu_thf.yaml` — old `.py` files deleted.

### Security hardening (post-review)
- **model_loader.py**: YAML/JSON only (no code execution), `.yaml`/`.yml`/`.json` extension enforced, 1 MB file size cap, schema validation
- **optimizer.py**: `ProcessPoolExecutor` capped at `MAX_WORKERS=8`, `MAX_PARAM_VALUES=200` limit, full tracebacks at DEBUG level, realization failure count warnings
- **instrument.py**: `MAX_DATA_FILE_SIZE=50MB` check before `np.loadtxt`
- **report.py**: `MAX_JSON_FILE_SIZE=100MB` check before `json.load`
- **cli.py**: `_validate_output_path()` rejects `..` in output dir; experiment/optimization bounds now enforced in model_loader validation functions rather than Click IntRange
- **experiment_design.py**: KDE fallback now logs full traceback at DEBUG level
- **No remaining code execution risk**: importlib fully removed

## Phase 2 — Text-to-Model Planning (LLM)

### Architecture
- **Plain-text input**: User provides a free-text description (.txt file) of their sample, hypothesis, and what to optimise. No structured query schema — the LLM infers everything (layer stack, SLD values, fit ranges, optimisation target, candidate values, instrument settings).
- **PlanQuery** (Pydantic): Single field `description: str` (min 10 chars, max 10000). `load_query()` accepts `.txt` (whole file) or `.yaml` (reads `description` key).
- **SLD database** (`sld_database.py`): Uses `periodictable` (bundled with refl1d) for neutron SLD computation. ~60 material aliases, ~20 compound densities, special-case air/vacuum = 0.0.
- **LLM pipeline**: Plain text → system prompt (YAML schema + SLD table) + user prompt → LangChain ChatOpenAI → YAML model → validator → retry loop (up to `max_retries`).
- **`.env` loading**: The CLI `main()` group calls `load_dotenv()` at startup so all subcommands pick up settings from `.env`. The `plan` and `plan-and-optimize` commands read `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MAX_TOKENS` from the environment (CLI flags override).
- **ALCF provider support**: When `LLM_PROVIDER=alcf`, `_get_llm_config()` automatically sets the base URL to the ALCF Sophia (or Metis) inference endpoint and falls back to `ALCF_ACCESS_TOKEN` for auth. `generate_model_yaml()` accepts `base_url` and `max_tokens` parameters.
- **Validator** (`validator.py`): Schema-level validation without building the full refl1d experiment. Checks layer structure, fit ranges, param references, experiment bounds. Accepts `description` as a top-level key. Fast enough for the LLM retry loop.
- **No code execution**: LLM generates declarative YAML, not Python. The generated output is validated and parsed with `yaml.safe_load` only.
- **`description` field in model YAML**: The LLM is instructed to include a `description` field in the generated model YAML. Both the validator and model_loader accept and ignore it. Example model YAMLs now include `description`.

### CLI commands
- `rose plan QUERY_FILE` — Generate a ROSE YAML model from a plain-text description via LLM. QUERY_FILE is a `.txt` or `.yaml` file. Options: `--output`, `--model-name`, `--temperature`, `--verbose`. Defaults for model/temperature come from `.env`.
- `rose plan-and-optimize QUERY_FILE` — Generate model then immediately run optimisation. Options include `--data-file` for measured data. Combines `plan` + `optimize`.
- `rose check-llm` — Check LLM configuration and connectivity. Shows provider, model, and credential status. For ALCF provider, shows Globus token availability instead of API key. For OpenAI/local, shows masked API key. Options: `--no-test`, `--json`, `--fix` (ALCF: download and run auth script). Reads `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` / `OPENAI_API_KEY`, `LLM_BASE_URL`, `LLM_TEMPERATURE`, `ALCF_ACCESS_TOKEN`, `ALCF_CLUSTER` from environment.

### Key design decisions
- **Plain text input (AuRE-style)**: The original structured query YAML (sample/layers/optimize/experiment) was simplified to plain text because the query was nearly as complex as the model YAML itself. Inspired by AuRE's approach where users write a paragraph and the LLM infers the full model.
- **YAML generation, not code generation**: The LLM outputs YAML model files (same schema as hand-authored Phase 1 models), not Python code. This eliminates code execution risk and leverages existing validation pipeline.
- **SLD reference table in prompt**: The system prompt includes a dynamically generated SLD reference table from the database so the LLM has accurate values.
- **Retry with error feedback**: If the LLM produces invalid YAML, the validator errors are fed back as a follow-up message and the LLM retries.
- **Markdown fence stripping**: LLMs often wrap output in ```yaml fences; `_strip_markdown_fences()` handles this.
- **ALCF Globus token authentication**: ALCF does not use API keys. Authentication uses Globus access tokens with 3-tier resolution: (1) `ALCF_ACCESS_TOKEN` env var, (2) `globus_sdk` UserApp if installed, (3) subprocess fallback to `inference_auth_token.py get_access_token`. The `_get_alcf_token()` function raises `RuntimeError` if all methods fail. `_alcf_token_available()` is a silent boolean check. ALCF cluster endpoints: sophia = `https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1`, metis = `https://inference-api.alcf.anl.gov/resource_server/metis/api/v1`. Auth helper script: `https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py`. The `plan` and `plan-and-optimize` commands resolve the token via `_get_alcf_token()` when provider is ALCF.
- **Reflection geometry awareness**: The system prompt guides the LLM to infer front vs back reflection from the measurement environment. "Measured in air/vacuum" → front reflection (air first, substrate last). "Measured against a liquid" (D₂O, THF, etc.) → back reflection (liquid first, substrate last). The user can override by explicitly stating the beam direction. This determines which layer is listed first in the generated YAML.

### Module structure
```
src/rose/modeler/
  __init__.py
  schema.py          — PlanQuery(description: str) + load_query()
  sld_database.py    — Material SLD lookup (periodictable)
  prompts.py         — System & user prompt templates
  llm_generator.py   — LangChain chain: description → YAML model
  validator.py       — YAML model schema validator
```

### Test coverage (Phase 2)
- 65 Phase 2 tests + 83 Phase 1 = 148 total, all passing
- Covers: SLD database (resolve, compute, lookup, list), query schema (description validation, .txt loading, .yaml loading, unsupported format, file not found), validator (valid models, missing keys, bad ranges, unknown layers), prompts, CLI help, markdown fence stripping, mocked LLM generation (success, retry, max-retries failure), check-llm (help, key present, key missing, JSON output, key masking, missing deps, ALCF token present, ALCF no token, ALCF JSON output)

### Dependencies
- `langchain>=0.3.0`, `langchain-openai>=0.2.0`, `pydantic>=2.4.0`, `python-dotenv>=1.0.0` in `[llm]` optional extras
- `python-dotenv>=1.0.0` also in `[cli]` extras
- `periodictable` comes transitively via `refl1d` ≥ 1.0.0
- `globus_sdk` optional — used for ALCF token resolution if installed
