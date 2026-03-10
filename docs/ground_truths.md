# Ground Truths

Key findings and verified facts for the ROSE project. This file is updated
as development progresses so that future decisions build on established knowledge.

## Architecture

- **Analyzer planner location**: `/Users/m2d/git/analyzer/analyzer_tools/planner/`
  - Core files: `experiment_design.py`, `optimizer.py`, `instrument.py`, `mcmc_sampler.py`, `report.py`, `cli.py`
  - Algorithm: Bayesian experimental design following Treece et al., J. Appl. Cryst. (2019), 52, 47–59
  - Two entropy methods: MVN (multivariate normal) and KDN (kernel density estimation)
  - Uses bumps DREAM sampler for MCMC posterior sampling

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
        1. Simulate noisy reflectivity from model
        2. Run MCMC (DREAM) on the noisy data
        3. Extract marginal posterior samples for parameters of interest
        4. Compute H_posterior via MVN or KDE
        5. ΔH = H_prior - H_posterior

    Average ΔH over realizations → information gain for this parameter value

Output: parameter value with maximum average ΔH
```

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
- `rose optimize --model-file ... --param ... --param-values ...` — runs full optimization
- `rose report --result-file ... --output-dir ...` — regenerates plots from JSON

### Test coverage (Phase 1 end)
- 68 tests, all passing (including 9 security tests)
- Modules tested: experiment_design (81%), instrument (91%), model_loader (96%), report (98%), cli (46%), core/types (100%), core/config (94%)
- `mcmc_sampler` and `optimizer` at 0% — these require MCMC which is slow; tested indirectly via CLI integration

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
- **cli.py**: `_validate_output_path()` rejects `..` in output dir, `IntRange(1,100)` on `--num-realizations`, `IntRange(100,100000)` on `--mcmc-steps`
- **experiment_design.py**: KDE fallback now logs full traceback at DEBUG level
- **No remaining code execution risk**: importlib fully removed
