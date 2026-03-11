# ROSE — Reflectometry Open Science Engine

Bayesian experiment design optimisation for neutron reflectometry.

ROSE helps scientists choose the best experimental conditions **before**
running a measurement. Given a model of the sample, it evaluates a grid
of controllable parameter values and reports which value maximises the
expected information gain about the parameters of interest.

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
```

## Use cases

| # | Description | Status |
|---|-------------|--------|
| 1 | Determine best experimental conditions from a known model | CLI ready (Phase 1) |
| 2 | Plan experiment from a textual description (LLM-assisted) | Planned (Phase 2) |

See [docs/use-case-1.md](docs/use-case-1.md) for a step-by-step guide to
use-case 1.

## Project structure

```
src/rose/
  cli.py                  Click CLI entry point
  core/
    config.py             Configuration (YAML + env vars)
    types.py              Shared dataclasses
  planner/
    model_loader.py       YAML/JSON → refl1d Experiment
    experiment_design.py  ExperimentDesigner (entropy, sampling)
    optimizer.py          Parallel/sequential optimisation loop
    instrument.py         Q-grid, noise simulation, data loading
    mcmc_sampler.py       DREAM MCMC via bumps
    report.py             PNG plot generation from results JSON
  modeler/                (Phase 2 — LLM model generation)
examples/
  models/
    layer_a_on_b.yaml     Simple two-layer example
    cu_thf.yaml           Cu/material/THF five-layer system
```

## Model file format

Models are described declaratively in YAML or JSON — no code execution
is involved. A single file contains three sections: the layer stack,
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

See [docs/use-case-1.md](docs/use-case-1.md) for a detailed explanation of the
schema.

## CLI commands

| Command | Purpose |
|---------|---------|
| `rose inspect <MODEL_FILE>` | Show variable and fixed parameters |
| `rose optimize <MODEL_FILE>` | Run Bayesian optimisation using settings from the file |
| `rose report --result-file ... --output-dir ...` | Regenerate plots from results JSON |

Run `rose --help` or `rose <command> --help` for full option details.

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

## License

BSD-3-Clause — see [LICENSE](LICENSE).
