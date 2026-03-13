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
- **Reflection geometry awareness**: The system prompt guides the LLM to always order layers from the free surface (air/liquid) to the substrate, regardless of beam direction. "Measured in air/vacuum" → air first, substrate last. "Measured against a liquid" (D₂O, THF, etc.) → liquid first, substrate last. The beam direction (front vs back reflection) does NOT change the layer ordering. This is because refl1d's `back_reflectivity` probe flag does not work reliably; instead, layers are always ordered from front surface to substrate, matching the aure project convention (see `/Users/m2d/git/aure/`). Example: Cu/Ti on Si measured in THF with beam from Si side → THF first, Si last.

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

## Phase 3 — Flask Web App for Visualization

### Architecture
- **Flask app factory**: `create_app(results_dir)` in `src/rose/web/__init__.py` — registers a single Blueprint, sets `RESULTS_DIR` config, `secret_key = "rose-web"`.
- **Blueprint pattern**: Single blueprint `bp` in `routes.py` with page routes and JSON API endpoints.
- **Data layer**: `ResultData` class in `data.py` — lazy-loads `optimization_results.json`, provides `get_summary()`, `get_info_gain()`, `get_reflectivity(index)`, `get_sld(index)`, `get_settings()`, `get_model_yaml()`. `list_results(results_dir)` scans for subdirectories containing the JSON file.
- **Path traversal protection**: `_find_result()` rejects `..`, `/`, `\` in result IDs.
- **AuRE visual design replicated**: Dark navbar (`bg-dark`, `navbar-dark`), `<i class="bi bi-layers"></i> ROSE` brand (no image logo, same as AuRE), light gray background (`#f8f9fa`), white cards with `box-shadow: 0 1px 3px rgba(0,0,0,0.08)`, no borders.
- **CDN dependencies**: Bootstrap 5.3.3, Bootstrap Icons 1.11.3, Plotly 2.35.0, all loaded from CDN.

### Routes
- **Page routes**: `GET /` (index listing), `GET /results/<id>` (detail with Plotly charts), `GET /results/<id>/model` (YAML viewer).
- **JSON APIs**: `GET /api/results` (summary list), `GET /api/results/<id>/info-gain`, `/reflectivity?index=N`, `/sld?index=N`, `/settings`, `/summary`.

### Templates
- `base.html` — Bootstrap 5.3 layout matching AuRE (dark navbar, `bi-layers` icon, Plotly CDN, `container-fluid`).
- `index.html` — Results listing with Bootstrap cards, summary stats per result, links to detail/model views, empty state with instructions.
- `result.html` — Detail page: breadcrumb nav, summary card row, interactive info gain chart (Plotly line+markers with error bars, star at optimal), parameter selector dropdown, R(Q) log-log chart (true + noisy reflectivity), SLD profile chart (with 90% CL bands), settings table. JavaScript fetches data from JSON APIs.
- `model.html` — Model YAML viewer: breadcrumb, YAML source in `<pre><code>`, settings table.

### CLI
- `rose serve RESULTS_DIR` — Start the Flask web app. `RESULTS_DIR` defaults to `results`. Options: `--port` (default 5000), `--no-browser`. Auto-opens browser unless `--no-browser`.

### Module structure
```
src/rose/web/
  __init__.py       — create_app() factory
  data.py           — ResultData class, list_results()
  routes.py         — Blueprint: page routes + JSON APIs
  templates/
    base.html       — AuRE-matching Bootstrap 5.3 layout
    index.html      — Results listing
    result.html     — Detail page with Plotly charts
    model.html      — Model YAML viewer
  static/
    style.css       — Minimal CSS overrides (AuRE-matching)
```

### Test coverage (Phase 3)
- 27 Phase 3 tests + 83 Phase 1 + Phase 2 = 110 total, all passing
- **TestPageRoutes** (8): index 200, shows result name/parameter, detail 200/shows parameter/404, model 200/404, path traversal rejected
- **TestAPIRoutes** (9): results list, info-gain, reflectivity (default/explicit/out-of-range), SLD, settings, summary, 404 for missing
- **TestResultData** (9): exists true/false, summary, info_gain, model_yaml present/missing, list_results/empty/nonexistent
- **TestServeCLI** (1): `rose serve --help` shows correct options

### Dependencies
- `flask>=3.0.0` in `[web]` extras (pyproject.toml)
- `plotly` CDN-loaded (no Python dependency needed — charts are rendered client-side via JSON API)

## Phase 4 — Interactive Web App + AuRE Plugin

### Architecture
- **Job management**: `JOBS` dict + `JOBS_LOCK` (`threading.Lock`) stored in `app.config`. Thread-safe updates via `_update_job()` helper. Pattern adopted from AuRE's `RUN_STATE` approach.
- **Background execution**: Daemon threads (`threading.Thread(daemon=True)`) for optimization and planning jobs. Each job gets a UUID-based ID (`uuid4()[:8]`).
- **Job lifecycle**: `POST /api/jobs/optimize` or `/api/jobs/plan` → validates inputs → creates job dict (`status: running`) → spawns daemon thread → returns `{job_id}`. Client polls `GET /api/jobs/<id>/status` every 2s. Job transitions: `running` → `complete` (with `result_dir`) or `error` (with message).
- **File browser**: Server-side directory listing following AuRE's pattern. `_safe_browse_path()` validates paths (must exist, must be a directory). Hidden files (starting with `.`) excluded.
- **AuRE plugin**: `register_with_aure(app, url_prefix="/rose")` registers the ROSE Blueprint with a URL prefix, initializes JOBS/JOBS_LOCK if not already present. Allows ROSE to be mounted inside AuRE at `/rose/`.
- **Form persistence**: Client-side `localStorage` saves form state (model file path, output dir, parallel toggle) across page reloads.
- **No new dependencies**: Phase 4 extends Phase 3's Flask app — no additional Python packages required.

### Routes (new in Phase 4)
- **Page routes**: `GET /optimize` (optimization form), `GET /plan` (LLM planning form)
- **File browser APIs**: `GET /api/browse-files?path=X&ext=.yaml` (files + dirs), `GET /api/browse-dirs?path=X` (dirs only)
- **Job APIs**: `POST /api/jobs/optimize`, `POST /api/jobs/plan`, `GET /api/jobs/<id>/status`

### Input validation
- **Optimize job**: Requires `model_file` (must exist, `.yaml`/`.yml`/`.json` extension) and `output_dir` (non-empty). Optional `data_file` (must exist if provided). Optional `parallel` boolean.
- **Plan job**: Requires `description` (min 10 chars) and `output_dir`. Optional `data_file`. The description is sent to the LLM to generate a model YAML, then optimization runs automatically.
- **File browser**: `_safe_browse_path()` rejects nonexistent paths and non-directories. Returns 400 with error message on invalid paths. `PermissionError` caught and returns empty entries.

### Background runners
- **`_run_optimize_job()`**: Loads model via `load_model_description()` + `build_experiment()`, creates `ExperimentDesigner`, runs `optimize_sequential()` or `optimize_parallel()`, saves results (JSON + plots), copies model YAML to output dir, updates `RESULTS_DIR`.
- **`_run_plan_job()`**: Reads LLM config from env (`LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_PROVIDER`), calls `generate_model_yaml()`, writes generated YAML to temp file, then delegates to `_run_optimize_job()`.
- **Error handling**: Both runners wrap execution in try/except, updating job status to `error` with the exception message on failure.

### Templates (new in Phase 4)
- `optimize.html` — Model file selector (with browse modal), optional data file, output directory, parallel toggle, progress panel (status badge, progress text, animated bar, "View Results" link).
- `plan.html` — Sample description textarea (with example placeholder), optional data file, output directory, parallel toggle, same progress panel pattern.
- `_browser_modal.html` — Shared Bootstrap modal partial: path breadcrumb, up-arrow navigation, file/folder listing, "Select this folder" button (dir mode). Included by both forms.

### Client-side (`setup.js`)
- **File browser**: `openBrowser(mode, ext)` → fetches listing from API → renders in modal → selection fills form field. Modes: `"file"` (model YAML), `"datafile"` (data file), `"dir"` (output directory).
- **Form persistence**: `localStorage` with key `"rose_setup"` — saves/restores model path, data path, output dir, parallel toggle.
- **Job polling**: `_pollStatus(jobId)` calls status API every 2s (3s on error). Updates progress bar, status badge, and text. On completion: shows "View Results" link. On error: shows error message.
- **Nav tabs**: Three tabs in dark navbar: Optimize (`bi-sliders`), Plan (`bi-pencil-square`), Results (`bi-folder2-open`). Active tab highlighted via `active_tab` template variable.

### Module structure (Phase 4 additions)
```
src/rose/web/
  __init__.py          — create_app() + register_with_aure()
  routes.py            — Extended: optimize/plan pages, file browser, job APIs, background runners
  templates/
    base.html          — Updated nav with Optimize/Plan/Results tabs
    optimize.html      — NEW: optimization setup form + progress panel
    plan.html          — NEW: LLM planning form + progress panel
    _browser_modal.html — NEW: shared file browser modal partial
  static/
    setup.js           — NEW: file browser, form persistence, job launch, polling
    style.css          — Unchanged
```

### Test coverage (Phase 4)
- 28 Phase 4 tests + 110 prior = **138 total, all passing**
- **TestInteractivePages** (5): optimize/plan page 200, form elements present, nav tabs
- **TestBrowseAPI** (10): browse-files (home, path, ext filter, nonexistent, path traversal, file-to-parent fallback, permission denied), browse-dirs (listing, parent), hidden files excluded
- **TestJobAPI** (10): optimize validation (no model, no output, bad data file), plan validation (short desc, no output), status 404, status returns state, internal keys hidden, happy-path optimize accepted, happy-path plan accepted
- **TestPluginRegistration** (3): register_with_aure default prefix, custom prefix, JOBS config initialized

### Deferred
- **MCP tool integration** (plan.md step 39): Deferred — requires FastMCP and is a separate integration concern. Can be added later when AuRE's MCP server is ready to consume ROSE tools.
- **Advanced form features**: Interactive parameter selection with checkboxes, sliders for value ranges, YAML syntax highlighting editor — deferred to future iterations. Current forms cover the essential workflow.

## Model Discrimination

### Motivation
Information gain alone can be misleading: a measurement condition may yield high ΔH for the primary model but fail to distinguish it from simpler alternatives. Example: in `cu_thf.yaml`, low THF SLD shows high information gain for CuOx parameters, but the data equally well explained by simple Cu roughness (no oxide layer). High THF SLD both informs CuOx parameters AND discriminates against the "no oxide" hypothesis.

### Approach
Inline alternate models defined in the YAML `optimization` section. Each alternate is a set of modifications (remove, modify, add) applied to the primary layer stack. For each realization, both the primary and all alternates are fit to the same noisy synthetic data, then a discrimination metric quantifies which model the data prefers.

### YAML Schema Extension
```yaml
optimization:
  alternate_models:
    - name: no_oxide
      modifications:
        - action: remove     # remove a layer
          layer: CuOx
        - action: modify     # change/fit a layer
          layer: Cu
          set: {interface: 15}
          fit: {interface: [3, 40]}
        - action: add        # add a new layer
          layer: {name: new, rho: 3.5, thickness: 10}
          after: Cu           # or before: Cu
  discrimination_method: bic     # "bic" or "evidence"
  discrimination_mode: report    # "report" or "penalize"
  alt_mcmc_steps: null           # defaults to mcmc_steps
```

### Discrimination Metrics
- **BIC**: `BIC = -2·logp_best + k·ln(n)`. `ΔBIC = BIC_alt - BIC_primary` (positive favours primary). `P(primary|data) = 1/(1+exp(-ΔBIC/2))`.
- **Evidence** (harmonic mean): Newton-Raftery estimator on DREAM chains. `log Z ≈ -(logsumexp(-logp) - log(N))`. Log Bayes factor = `log Z_primary - log Z_alt`. `P(primary|data) = 1/(1+exp(-log_bf))`. High variance but requires no additional MCMC runs.
- **bumps limitation**: bumps DREAM has no built-in log-evidence or thermodynamic integration. Harmonic mean estimator uses `state.sample(portion=0.3)` which returns `(points, logp)`.

### Combined Scoring
- **report mode**: Both info gain and P(primary) reported side-by-side. No modification to optimization recommendation.
- **penalize mode**: `effective_info_gain = info_gain × mean(P(primary|data))`. Conditions where the primary can't be distinguished from alternates get penalized. Optimal parameter selected by max effective_info_gain.

### Module Structure
```
src/rose/planner/
  model_discriminator.py  — ModelDiscriminator class, model_probability(), combine_scores()
  experiment_design.py    — compute_bic(), compute_log_evidence() (appended)
  model_loader.py         — _validate_alternate_models(), build_alternate_experiments()
  optimizer.py            — discriminator threaded through evaluate_param/optimize/optimize_parallel
```

### Data Flow
1. CLI loads YAML → `_validate_alternate_models()` checks schema + layer references
2. `build_alternate_experiments()` deep-copies primary desc, applies modifications, builds refl1d Experiments
3. `ModelDiscriminator(alt_experiments, method)` created from list of `(name, Experiment)` tuples
4. Passed through `optimize()`/`optimize_parallel()` → `evaluate_param()` / `_evaluate_single_realization()`
5. Per realization: alternates are fit to same noisy data via `perform_mcmc()`, metrics computed
6. CLI aggregates per-value discrimination → `combine_scores()` → JSON output
7. `report.py` generates `model_discrimination.png` (twin-axis: P(primary) + ΔH)

### Output
- JSON: `"discrimination"` key with `alternate_models`, `method`, `mode`, and `per_value` array (each with `mean_model_prob`, `mean_delta_metric`, `info_gain`, `mean_model_prob` scalar, optionally `effective_info_gain`)
- Plot: `model_discrimination.png` — P(primary|data) per alternate on left axis, ΔH on right axis. Penalize mode also shows effective ΔH.
- Plot: `information_gain.png` — in penalize mode, overlays faded-blue raw ΔH (with error bars) and bold-green penalized ΔH. Standard mode unchanged.
- CLI table: penalize mode adds `Eff. ΔH` column alongside `Value`, `ΔH (bits)`, `± std`, `P(primary)`.
- ASCII graph: penalize mode shows blue `━` bars for raw ΔH and green `━` bars for penalized ΔH (ANSI colours).

### Test Coverage
- 176 total tests, all passing
- Model discrimination tests in `test_model_discriminator.py`: model_probability, combine_scores, ModelDiscriminator construction, `_set_param_on_sample` (3 tests)
- Alternate model validation tests in `test_model_loader.py`: valid/invalid actions, missing name, unknown layer, bad discrimination_method/mode; build_alternate_experiments: remove, modify, deep-copy safety, multiple alternates

### Parameter-on-alternates fix (2026-03-12)
- **Bug**: `designer.set_parameter_to_optimize(param, value)` only sets the parameter on the primary model. Alternate experiments (built once at startup from YAML modifications) retain their YAML default values. When the optimizer sweeps `THF rho` from 0 to 7, the alternate model always uses `THF rho = 5.8` (the YAML initial), making discrimination results wrong.
- **Fix**: `_set_param_on_sample(experiment, name, value)` helper walks the `FitProblem(experiment)._models[0].parameters()["sample"]["layers"]` tree to find a `Parameter` object matching `name` (format: `"{layer_name} {property}"`) and directly sets `sub_param.value = value`. Since `FitProblem(experiment)` references the same `Parameter` objects as the experiment's sample, the change propagates to the actual experiment used by `perform_mcmc()`.
- **Graceful handling**: Returns `False` (with `logger.debug`) when the named parameter doesn't exist in the alternate (e.g., after a `remove` action deletes the layer containing that parameter). This is expected and correct — the alternate's structure simply doesn't include that layer.
- **Call site**: `ModelDiscriminator.evaluate()` accepts `param_to_optimize: str | None` and `param_value: float | None`. Before each alternate's MCMC run, calls `_set_param_on_sample()` to apply the current optimisation condition.

## FitProblem Serialization (bumps/refl1d)

### Key Finding: bumps provides full JSON serialization of FitProblem objects

There are **three distinct mechanisms** for saving/loading FitProblem state:

### 1. Full JSON Serialization via `bumps.serialize` (PREFERRED for programmatic use)

**Save:**
```python
from bumps.serialize import serialize, save_file
import json

# Option A: serialize to dict, then dump
serialized = serialize(problem)  # returns dict with $schema, object, references keys
with open("problem.json", "w") as f:
    json.dump(serialized, f)

# Option B: convenience function (wraps Option A)
save_file("problem.json", problem)
```

**Load:**
```python
from bumps.serialize import deserialize, load_file

# Option A: load from dict
import json
with open("problem.json", "r") as f:
    serialized = json.loads(f.read())
problem = deserialize(serialized, migration=True)

# Option B: convenience function
problem = load_file("problem.json")
```

**Format:** JSON with schema versioning (`bumps-draft-03`). Uses `__class__` keys for type info,
`$ref` references for shared objects (parameters), base64-encoded numpy arrays, and
cloudpickle-serialized callables. Supports schema migrations across versions.

**Location:** `bumps.serialize` module (functions: `serialize`, `deserialize`, `save_file`, `load_file`)

### 2. Higher-level serialize/deserialize (webview server API)

```python
from bumps.webview.server.state_hdf5_backed import serialize_problem, deserialize_problem

# Supports methods: "dataclass" (JSON), "pickle", "cloudpickle", "dill"
serialized_str = serialize_problem(problem, method="dataclass")  # returns JSON string
problem = deserialize_problem(serialized_str, method="dataclass")

# Also: serialize_problem_bytes / deserialize_problem_bytes for bytes output
```

Default serializer is `"dataclass"` (JSON). Fallback options: `"pickle"`, `"dill"`, `"cloudpickle"`.

### 3. Parameter-only `.par` file (used by CLI fitting)

**Save (in `bumps.cli.save_best`):**
```python
# Writes label-value pairs to .par file
pardata = "".join("%s %.15g\n" % (name, value)
                  for name, value in zip(problem.labels(), problem.getp()))
open(output_path + ".par", "wt").write(pardata)
```

**Load (in `bumps.cli.load_best`):**
```python
from bumps.cli import load_best
load_best(problem, "path/to/results.par")  # updates parameter values in existing problem
```

**Format:** Plain text, one "label value" per line. Only saves parameter values, NOT the full model structure. Requires an existing FitProblem to load into.

### 4. Python script loading (`bumps.fitproblem.load_problem`)

```python
from bumps.fitproblem import load_problem
problem = load_problem("model_script.py", options=[])
```

Executes a Python script that must define `problem = FitProblem(...)`. This is how `.py` model files are loaded (the pattern used in `results/problems/`).

### Gotchas & Limitations

- **JSON serialization requires dataclass-based models**: Objects need `__schema__` attribute and dataclass decorators for clean serialization. refl1d models are dataclasses and support this.
- **Callable constraints**: If `constraints` is a function (not constraint expressions), it gets cloudpickle-serialized as base64 — works but fragile across Python versions.
- **`save()` on FitProblem**: The `save(basename)` method delegates to each model's `save()`. For refl1d, this saves profiles (.dat), reflectivity (.dat), and experiment JSON (.json) — NOT the full FitProblem structure. It's for result output, not for serialization.
- **refl1d Experiment.save_json()**: Saves just the experiment (fitness) as JSON via `bumps.serialize.serialize(self)`, NOT the full FitProblem wrapper.
- **Schema migrations**: `bumps.serialize` supports versioned schemas with automatic migration. Current version is `bumps-draft-03`.
- **load_problem() uses exec()**: The `.py` file loader uses `exec()` — only use with trusted files.

### ROSE Integration

ROSE uses `bumps.serialize.serialize()` to capture the exact `FitProblem` used during MCMC.
When `--save-problems` is passed to `rose optimize`:

1. `perform_mcmc()` returns `(result, problem)` — the problem is the actual `FitProblem` with noisy data.
2. The optimizer serializes the problem via `bumps.serialize.serialize()` into `rdata["problem"]` (first realization only).
3. Alternate model problems are serialized into `rdata["alt_problems"]` via `ModelDiscriminator.evaluate()`.
4. The CLI writes these as JSON files to `{output_dir}/problems/step_{i}_value_{v}_{label}.json`.
5. Files can be reloaded via `bumps.serialize.load_file(path)` for inspection or further fitting.

## Removed Features

### `export_model_script()` (removed 2026-03-12)
- Originally generated standalone refl1d Python model files from YAML descriptions for human inspection.
- Replaced by `bumps.serialize` which provides full `FitProblem` JSON serialization with actual fitted data, parameters, and state.
- The `_py_varname()` helper was also removed as it was only used by `export_model_script()`.

### `export_problem_script()` (removed 2026-03-12)
- Originally generated Python scripts with embedded numpy arrays for data.
- Replaced by `bumps.serialize` (see above) which captures exact FitProblem objects.
