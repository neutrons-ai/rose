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
- **Remaining CLI options**: `MODEL_FILE` (positional), `--data-file`, `--output-dir`, `--parallel/--sequential`, `--verbose`.
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
