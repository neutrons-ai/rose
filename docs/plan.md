# ROSE Implementation Plan

**TL;DR**: Build ROSE across 5 phases — port the analyzer planner's Bayesian information-gain algorithm into a clean package, add an LLM-powered text-to-model pipeline, then wrap it all in a Flask web app designed to also serve as an AuRE plugin.

## Decisions

- **Scope**: All phases (0–4)
- **Use-case 2 approach**: LLM-based (LangChain) for text → refl1d model generation
- **Planner code strategy**: Copy & adapt from `analyzer_tools/planner/` into ROSE (not a dependency on analyzer)
- **Tech stack**: Click CLI, Flask web app, FastAPI for future MCP, pytest for testing

---

## Phase 0: Package Structure

**Goal**: Establish the module layout, dependencies, dev tooling, and CI foundation.

### Steps

1. **Create module layout** under `src/rose/`:
   - `core/` — shared types, config, utilities
     - `config.py` — YAML/dotenv configuration loading
     - `types.py` — shared data classes (OptimizationResult, SampleModel, etc.)
   - `planner/` — experiment design engine (ported from analyzer)
   - `modeler/` — text-to-model LLM pipeline (use-case 2)
   - `cli.py` — Click CLI (already exists, will be extended)
   - `web/` — Flask web app (phase 3+)

2. **Update `pyproject.toml` dependencies**:
   - Add core: `refl1d`, `bumps`, `scipy`, `pyyaml`
   - Add `llm` optional extra: `langchain`, `langchain-openai`, `openai`
   - Keep existing optional groups (`dev`, `web`, `cli`, `science`)
   - Add `[project.scripts]` entry point: `rose = "rose.cli:main"`

3. **Add dev tooling configuration**:
   - `ruff.toml` or `[tool.ruff]` in pyproject.toml
   - `[tool.pytest.ini_options]` configuration
   - `[tool.mypy]` configuration
   - `.pre-commit-config.yaml`

4. **Create `docs/ground_truths.md`** for tracking key findings

5. **Add basic CI** (GitHub Actions): lint + test on push

### Relevant Files

- `pyproject.toml` — add dependencies, entry points, tool config
- `src/rose/__init__.py` — update exports
- `src/rose/core/` — new directory (types.py, config.py)
- `.github/workflows/ci.yml` — new CI workflow

### Verification

- `pip install -e ".[dev,cli]"` succeeds
- `rose --help` shows CLI
- `pytest` passes (existing tests still work)
- `ruff check src/` passes

---

## Phase 1: CLI for Use-Case 1 (Experiment Optimization)

**Goal**: Port the analyzer planner into ROSE and expose it via CLI. Given a refl1d model, optimize controllable parameters to maximize information gain.

### 1A — Port Core Engine (5 steps)

6. **Port `experiment_design.py`** → `src/rose/planner/experiment_design.py`
   - `ExperimentDesigner` class with prior/posterior entropy calculation
   - Support both MVN and KDE entropy methods
   - Accept `parameters_of_interest` for marginalized distributions
   - Adapt to use ROSE's own type classes

7. **Port `instrument.py`** → `src/rose/planner/instrument.py`
   - `InstrumentSimulator` class for noise simulation
   - Support loading Q/dQ from data files or generating ranges

8. **Port `mcmc_sampler.py`** → `src/rose/planner/mcmc_sampler.py`
   - `perform_mcmc()` using bumps DREAM sampler
   - Configurable burn-in, steps, and population size

9. **Port `optimizer.py`** → `src/rose/planner/optimizer.py`
   - `optimize()` — sequential evaluation across parameter values
   - `optimize_parallel()` — ProcessPoolExecutor-based parallelism
   - `evaluate_param()` — single parameter value evaluation workhorse

10. **Port `report.py`** → `src/rose/planner/report.py`
    - Information gain curve plotting
    - Reflectivity plots with simulated data
    - SLD depth profiles with uncertainty bands (90% CL)

### 1B — Model Loading (2 steps)

11. **Create `src/rose/planner/model_loader.py`**
    - Load refl1d model from Python module (`.py` file with `create_fit_experiment`)
    - Validate model has required parameters and structure
    - Extract parameter metadata (names, bounds, current values)

12. **Create example model files** under `examples/models/`
    - `layer_a_on_b.py` — simple two-layer system from project.md
    - `cu_thf.py` — adapted from analyzer example

### 1C — CLI Commands (3 steps)

13. **Extend `src/rose/cli.py`** with Click group and `optimize` subcommand:
    ```
    rose optimize \
      --model-file models/example.py \
      --param "layer_b thickness" \
      --param-values "10,20,30,40,50" \
      --parameters-of-interest "layer_a thickness" \
      --num-realizations 3 \
      --mcmc-steps 2000 \
      --entropy-method kdn \
      --output-dir results/ \
      --parallel
    ```

14. **Add `rose inspect` subcommand**: Display model parameters, bounds, and current values

15. **JSON + plot output**: Save `optimization_results.json` and PNG plots to `--output-dir`

### 1D — Tests (3 steps)

16. **Unit tests** for planner modules: `tests/planner/test_experiment_design.py`, `test_optimizer.py`, `test_instrument.py`
    - Test entropy calculations with known distributions
    - Test noise simulation statistics
    - Test parameter extraction and setting
    - Mock MCMC for fast tests

17. **Integration test**: `tests/planner/test_integration.py`
    - End-to-end: model file → optimize → JSON output
    - Use minimal MCMC steps for speed

18. **CLI tests**: `tests/test_cli.py`
    - Test `rose optimize --help`
    - Test `rose optimize` with example model
    - Test `rose inspect` output

### Relevant Files

- `src/rose/planner/` — new package (experiment_design.py, optimizer.py, instrument.py, mcmc_sampler.py, report.py, model_loader.py)
- `src/rose/cli.py` — extend with `optimize` and `inspect` commands
- `examples/models/` — example refl1d model files
- `tests/planner/` — new test directory
- Reference: analyzer repo `analyzer_tools/planner/` — source to port from

### Verification

- `rose inspect examples/models/layer_a_on_b.py` lists parameters
- `rose optimize --model-file examples/models/layer_a_on_b.py --param "layer_b thickness" --param-values "10,20,30" --num-realizations 1 --mcmc-steps 500 --output-dir /tmp/rose_test` produces JSON + plots
- `pytest tests/planner/` — all pass
- Information gain values for known setups match analyzer output

---

## Phase 2: CLI for Use-Case 2 (Text-to-Model Planning)

**Goal**: From a YAML text description of sample geometry and hypothesis, generate a refl1d model using an LLM, then feed it into the Phase 1 optimizer.

### 2A — YAML Input Schema (2 steps)

19. **Define `query.yaml` schema** in `src/rose/modeler/schema.py`
    - Pydantic models for: sample description (layers, materials, SLDs), hypothesis, controllable parameters, instrument config
    - Example:
      ```yaml
      sample:
        description: "Layer A on top of layer B on silicon substrate"
        layers:
          - name: "Layer A"
            material: "polymer"
            thickness_range: [10, 100]
          - name: "Layer B"
            material: "gold"
            thickness: 50  # controllable
        substrate: "silicon"
      hypothesis: "Find best thickness of Layer B to be sensitive to Layer A thickness"
      optimize:
        controllable: ["Layer B thickness"]
        parameters_of_interest: ["Layer A thickness"]
      instrument:
        q_range: [0.008, 0.2]
        q_points: 200
      ```

20. **Create example query files** under `examples/queries/`
    - `layer_sensitivity.yaml` — matches the project.md example
    - `thin_film.yaml` — simple thin film characterization

### 2B — LLM Model Generation (3 steps)

21. **Create `src/rose/modeler/llm_generator.py`**
    - LangChain chain that takes YAML input → generates refl1d Python model code
    - System prompt with refl1d API examples, SLD database context
    - Output validation: generated code must define `create_fit_experiment`
    - Sandboxed execution to validate the generated model runs

22. **Create `src/rose/modeler/sld_database.py`**
    - Material SLD lookup table (common materials for reflectometry)
    - Can also import from AuRE's database if available
    - Used as context for LLM to generate accurate SLD values

23. **Create `src/rose/modeler/validator.py`**
    - Load and execute generated model in sandbox
    - Verify it produces valid reflectivity curves
    - Check parameter ranges are reasonable
    - Return validation report to user

### 2C — CLI Commands (2 steps)

24. **Add `rose plan` subcommand**:
    ```
    rose plan \
      --query examples/queries/layer_sensitivity.yaml \
      --output-model generated_model.py \
      --llm-model gpt-4 \
      --validate
    ```
    - Generates model, validates it, saves to file
    - Shows generated model to user for approval

25. **Add `rose plan-and-optimize` combined command**:
    ```
    rose plan-and-optimize \
      --query examples/queries/layer_sensitivity.yaml \
      --num-realizations 3 \
      --output-dir results/
    ```
    - Runs `plan` → user reviews model → `optimize`
    - Interactive: pauses for user confirmation between steps

### 2D — Tests (2 steps)

26. **Unit tests** for modeler: `tests/modeler/test_schema.py`, `test_validator.py`, `test_sld_database.py`
    - Schema validation with valid/invalid YAML
    - Model validation with known good/bad models
    - SLD lookups

27. **Integration test with mocked LLM**: `tests/modeler/test_llm_generator.py`
    - Mock LLM responses with known good model code
    - End-to-end: YAML → generate → validate → optimize

### Relevant Files

- `src/rose/modeler/` — new package (schema.py, llm_generator.py, sld_database.py, validator.py)
- `src/rose/cli.py` — add `plan` and `plan-and-optimize` commands
- `examples/queries/` — YAML query files
- Reference: AuRE repo `src/aure/nodes/` — LangGraph pipeline patterns
- Reference: AuRE repo `src/aure/database/` — SLD database to reuse

### Verification

- `rose plan --query examples/queries/layer_sensitivity.yaml --output-model /tmp/test_model.py --validate` generates a valid model file
- `rose plan-and-optimize --query examples/queries/layer_sensitivity.yaml --num-realizations 1 --output-dir /tmp/rose_test2` runs end-to-end
- `pytest tests/modeler/` — all pass

---

## Phase 3: Flask Web App for Visualization

**Goal**: Web interface to visualize optimization results from use-cases 1 and 2. Read-only display of results.

### Steps

28. **Create `src/rose/web/__init__.py`** with `create_app()` factory
    - Blueprint-based structure (matching AuRE's pattern)
    - Bootstrap-Flask for responsive styling
    - Jinja2 templates

29. **Create `src/rose/web/routes.py`** with blueprint:
    - `GET /` — Landing page with links to browse results
    - `GET /results` — List available result directories
    - `GET /results/<id>` — View specific optimization result
    - `GET /api/results/<id>` — JSON API for result data (for JS plots)

30. **Add `rose serve` CLI command**:
    ```
    rose serve --results-dir results/ --port 5000
    ```

31. **Create result viewer template** (`templates/result.html`):
    - Information gain curve (interactive, Plotly)
    - Reflectivity plots per parameter value
    - SLD depth profiles with uncertainty bands
    - Summary table: optimal value, max info gain, settings

32. **Create model viewer** (`templates/model.html`):
    - Display model structure (layers, parameters, bounds)
    - Show generated model code (for use-case 2 results)
    - Interactive parameter table

33. **Create results listing template** (`templates/results.html`):
    - Browse past optimization runs
    - Filter by date, model type, parameter
    - Quick summary cards

34. **Flask test client tests**: `tests/web/test_routes.py`
    - Test all routes return 200
    - Test API returns valid JSON
    - Test with sample result fixtures

### Relevant Files

- `src/rose/web/` — new package (routes.py, templates/, static/)
- `src/rose/cli.py` — add `serve` command
- Reference: AuRE repo `src/aure/web/` — Flask patterns to follow

### Verification

- `rose serve --results-dir examples/sample_results/` starts server
- Browse to localhost:5000 → see results list
- Click result → interactive plots render
- `pytest tests/web/` — all pass

---

## Phase 4: Interactive Flask App + AuRE Plugin

**Goal**: Extend the web app to interactively enter inputs for use-cases 1 and 2. Design for standalone use but structured to be mountable as an AuRE blueprint.

### 4A — Interactive Input Forms (3 steps)

35. **Create optimization form** (`templates/optimize.html`):
    - Upload or select model file
    - Interactive parameter selection (checkboxes for which to optimize)
    - Slider/inputs for parameter value ranges
    - Settings: realizations, MCMC steps, entropy method
    - Submit → runs optimization as background task

36. **Create planning form** (`templates/plan.html`):
    - YAML editor (textarea with syntax highlighting) or structured form
    - Sample geometry builder (add/remove layers, set materials from SLD database)
    - Hypothesis text input
    - "Generate Model" button → shows generated model for review
    - "Approve & Optimize" button → runs pipeline

37. **Add background task execution**:
    - Thread-based task runner (similar to AuRE's pattern)
    - SSE or polling endpoint for progress updates
    - `POST /api/optimize` — start optimization job
    - `GET /api/jobs/<id>/status` — poll progress
    - `GET /api/jobs/<id>/result` — get completed result

### 4B — AuRE Plugin Structure (2 steps)

38. **Structure ROSE web as a standalone Blueprint**:
    - `src/rose/web/blueprint.py` — self-contained blueprint with `url_prefix="/rose"`
    - All templates namespaced under `rose/`
    - Static assets under `rose/static/`
    - `register_with_aure(app)` function that registers the blueprint

39. **Create MCP tool integration** in `src/rose/mcp_tools.py`:
    - `@mcp.tool()` decorated functions for AuRE's MCP server
    - `optimize_experiment()` — run use-case 1 from MCP
    - `plan_experiment()` — run use-case 2 from MCP
    - Can be imported and registered in AuRE's mcp_server.py

### 4C — Tests (1 step)

40. **Interactive feature tests**: `tests/web/test_interactive.py`
    - Test form submission
    - Test job creation and status polling
    - Test blueprint registration standalone and with mock AuRE app

### Relevant Files

- `src/rose/web/routes.py` — extend with interactive endpoints
- `src/rose/web/templates/` — new templates (optimize.html, plan.html)
- `src/rose/web/blueprint.py` — AuRE-mountable blueprint
- `src/rose/mcp_tools.py` — MCP tool definitions
- Reference: AuRE repo `src/aure/web/routes.py` — Flask UI patterns
- Reference: AuRE repo `src/aure/mcp_server.py` — MCP tool patterns

### Verification

- Submit optimization via web form → see live progress → view results
- Blueprint registers in test AuRE app with `app.register_blueprint(rose_bp, url_prefix="/rose")`
- `pytest tests/web/` — all pass

---

## Cross-Cutting Concerns

- **Documentation**: Update `README.md` at each phase with usage examples. Maintain `docs/ground_truths.md` with key findings.
- **Type hints**: All public functions must have type annotations
- **Docstrings**: Google-style on all public APIs
- **Error messages**: Clear, actionable messages for scientists (not developer jargon)
- **Logging**: Use Python `logging` module throughout, with `--verbose` CLI flag

## Dependency Summary

| Phase | New Dependencies |
|-------|-----------------|
| 0 | (existing) numpy, pandas + ruff, pytest, black |
| 1 | refl1d, bumps, scipy, matplotlib, click, pyyaml |
| 2 | langchain, langchain-openai, openai (optional `llm` extra) |
| 3 | flask, bootstrap-flask, plotly, jinja2 |
| 4 | (no new deps — extends phase 3) |

## Further Considerations

1. **MCMC performance**: Full MCMC runs are slow. Consider adding a `--quick` mode with reduced steps for prototyping, and document expected runtimes for scientists.
2. **LLM API key management**: Use-case 2 requires API keys. Use `python-dotenv` with `.env` file and document setup clearly. Consider supporting local models as fallback.
3. **AuRE version compatibility**: The AuRE plugin integration (Phase 4) should be tested against a specific AuRE version. Pin the integration target early.