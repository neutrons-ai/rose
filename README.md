# ROSE — Reflectometry Open Science Engine

Bayesian experiment design optimisation for neutron reflectometry.

ROSE helps scientists choose the best experimental conditions **before**
running a measurement.  Given a model of the sample, it evaluates a grid
of controllable parameter values and reports which value maximises the
expected information gain about the parameters of interest.  It can also
compare alternate structural hypotheses (model discrimination) to find
conditions that best distinguish between competing models.

## Quick start

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Inspect a model to see its variable and fixed parameters
rose inspect examples/models/layer_a_on_b.yaml

# Run optimisation — everything is configured in the YAML file
rose optimize examples/models/layer_a_on_b.yaml

# Custom output directory
rose optimize examples/models/layer_a_on_b.yaml --output-dir my_results

# Save refl1d FitProblem files for inspection
rose optimize examples/models/cu_thf.yaml --save-problems
```

## Use cases

| # | Description | Status |
|---|-------------|--------|
| 1 | Determine best experimental conditions from a known model | CLI + web app |
| 2 | Plan experiment from a textual description (LLM-assisted) | CLI + web app |

- [Use-case 1 guide](docs/use-case-1.md) — step-by-step walkthrough
- [Use-case 2 guide](docs/use-case-2.md) — LLM-powered text-to-model planning

## Project structure

```
src/rose/
  cli.py                    Click CLI entry point
  core/
    config.py               Configuration (YAML + env vars)
    types.py                Shared dataclasses
  planner/
    model_loader.py         YAML/JSON → refl1d Experiment
    experiment_design.py    ExperimentDesigner (entropy, sampling)
    optimizer.py            Parallel/sequential optimisation loop
    instrument.py           Q-grid, noise simulation, data loading
    mcmc_sampler.py         DREAM MCMC via bumps
    model_discriminator.py  Alternate model comparison (BIC / evidence)
    report.py               PNG plot generation from results JSON
  modeler/
    llm_generator.py        LangChain LLM → YAML model pipeline
    prompts.py              System and user prompt templates
    schema.py               PlanQuery schema + query loading
    sld_database.py         Material SLD lookup (periodictable)
    validator.py            YAML model schema validator
  web/
    __init__.py             Flask app factory + AuRE plugin hook
    data.py                 ResultData loader
    routes.py               Page routes + JSON APIs + background jobs
    templates/              Jinja2 templates (Bootstrap 5.3 + Plotly)
    static/                 CSS + JS
examples/
  models/
    layer_a_on_b.yaml       Simple two-layer example
    cu_thf.yaml             Cu/CuOx/THF with model discrimination
    copper_oxide.yaml       Copper oxide on silicon
    cu_ionomer.yaml         Cu with ionomer overlayer
```

## Model file format

Models are described declaratively in YAML or JSON — no code execution
is involved.  A single file contains three sections: the layer stack,
instrument settings, and optimisation configuration:

```yaml
name: Layer A on Layer B

layers:
  - name: air
    rho: 0.0
  - name: layer_A
    rho: 3.0
    thickness: 40
    interface: 8
    fit:
      thickness: [10, 100]
      rho: [1.0, 5.0]
  - name: layer_B
    rho: 4.5
    thickness: 50
  - name: Si
    rho: 2.07

experiment:
  q_min: 0.008
  q_max: 0.2
  q_points: 50
  dq_over_q: 0.025
  relative_error: 0.10

optimization:
  param: layer_B thickness
  param_values: [20, 40, 60, 80]
  parameters_of_interest: [layer_A thickness]
  num_realizations: 5
  mcmc_steps: 3000
  entropy_method: kdn
```

### Model discrimination

When alternate structural hypotheses exist, define them inline in the
`optimization` section.  ROSE fits each alternate to the same synthetic
data and computes a discrimination metric:

```yaml
optimization:
  # ... (param, param_values, etc.)
  discrimination_method: bic      # "bic" or "evidence"
  discrimination_mode: penalize   # "report" or "penalize"
  alternate_models:
    - name: no_oxide
      modifications:
        - action: remove
          layer: CuOx
        - action: modify
          layer: Cu
          fit:
            thickness: [500, 600]
```

- **report** mode: shows $P(\text{primary} | \text{data})$ alongside $\Delta H$.
- **penalize** mode: computes $\Delta H_{\text{eff}} = \Delta H \times \bar{P}(\text{primary})$,
  downweighting conditions where the primary model cannot be distinguished
  from alternatives.

See [docs/use-case-1.md](docs/use-case-1.md) for the full schema reference.

## CLI commands

| Command | Purpose |
|---------|---------|
| `rose inspect <MODEL>` | Show variable and fixed parameters |
| `rose optimize <MODEL>` | Run Bayesian optimisation (all settings from YAML) |
| `rose report --result-file ... --output-dir ...` | Regenerate plots from results JSON |
| `rose plan <QUERY_FILE>` | Generate a YAML model from a text description via LLM |
| `rose plan-and-optimize <QUERY_FILE>` | Generate model then run optimisation |
| `rose check-llm` | Verify LLM configuration and connectivity |
| `rose serve [RESULTS_DIR]` | Launch the Flask web app for interactive use |

Key `optimize` options:

| Option | Default | Description |
|--------|---------|-------------|
| `--data-file` | *(none)* | Measurement file — overrides Q-grid from YAML |
| `--output-dir` | `results` | Where to save JSON and plots |
| `--parallel / --sequential` | parallel | Use multiprocessing or not |
| `--workers N` | auto | Max parallel workers (default: min(tasks, CPUs, 8)) |
| `--save-problems` | off | Export refl1d FitProblem JSON files (via `bumps.serialize`) |
| `--verbose` | off | Debug-level logging |

Run `rose --help` or `rose <command> --help` for full details.

## Web app

```bash
# Install with web extras
pip install -e ".[dev,web]"

# Start the server (opens browser automatically)
rose serve results/
```

The web app provides interactive Plotly charts for information gain,
reflectivity, SLD profiles, and model discrimination.  It can also
launch optimisation and LLM planning jobs from the browser.

## Output files

| File | Description |
|------|-------------|
| `optimization_results.json` | Full results with per-realisation data |
| `information_gain.png` | ΔH vs. parameter value (+ penalized ΔH if applicable) |
| `simulated_data_<i>.png` | Reflectivity curves per candidate value |
| `sld_contours_<i>.png` | SLD profiles with 90% confidence bands |
| `model_discrimination.png` | P(primary) + ΔH twin-axis plot (if alternates defined) |
| `problems/*.json` | Serialised FitProblem files (with `--save-problems`) |

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Algorithm

ROSE implements the Bayesian experimental design approach from
[Treece et al., J. Appl. Cryst. (2019), 52, 47–59](https://doi.org/10.1107/S1600576718017016).
For each candidate value of the controllable parameter it simulates noisy
reflectivity data, runs MCMC (bumps DREAM) to obtain the posterior, and
computes the information gain $\Delta H = H_{\text{prior}} - H_{\text{posterior}}$.
The value with the highest average $\Delta H$ is optimal.

When alternate models are configured, each alternate is also fit to the
same synthetic data using its own MCMC run.  The discrimination metric
(BIC difference or log Bayes factor) quantifies whether the data
supports the primary model over each alternative.

## License

BSD-3-Clause — see [LICENSE](LICENSE).
