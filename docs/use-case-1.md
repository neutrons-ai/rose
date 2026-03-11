# Use-Case 1: Determining Best Experimental Conditions

This guide walks you through the full workflow for use-case 1 — choosing
the optimal value of a controllable parameter (e.g. a film thickness) so
that a neutron reflectometry measurement will be maximally sensitive to
the parameters you care about.

## Prerequisites

```bash
# Install ROSE in editable mode
pip install -e ".[dev]"

# Verify the CLI is available
rose --help
```

ROSE depends on **refl1d** (≥ 1.0.0) and **bumps** (installed transitively
with refl1d).  If you are working in a fresh environment you may need to
install refl1d first:

```bash
pip install refl1d
```

## Overview

The workflow has three steps:

1. **Describe the model** — write a YAML (or JSON) file listing the
   layer stack, instrument settings, and optimisation configuration.
2. **Inspect the model** — confirm which parameters ROSE will treat as
   variable (fitted) vs. fixed.
3. **Run the optimisation** — a single command reads everything from the
   model file and reports the optimal value.

```
┌─────────────┐      ┌──────────┐      ┌──────────┐
│  Write YAML │ ───▶ │  Inspect │ ───▶ │ Optimize │
│  model file │      │  params  │      │  & report│
└─────────────┘      └──────────┘      └──────────┘
```

---

## Step 1: Write the model file

A model file describes the sample as a stack of layers from top
(incident medium, e.g. air) to bottom (substrate, e.g. silicon),
plus instrument settings and what to optimise.
ROSE supports `.yaml`, `.yml`, and `.json` formats.

### YAML schema

```yaml
# Optional human-readable name
name: Layer A on Layer B

# ── Layer stack (top to bottom) ──────────────────────────
layers:
  - name: air             # incident medium — always first
    rho: 0.0              # SLD in 10⁻⁶ Å⁻²
    thickness: 0          # semi-infinite (omit or set to 0)
    interface: 5          # interfacial roughness (Å)

  - name: layer_A
    rho: 3.0
    thickness: 40         # initial thickness (Å)
    interface: 8
    fit:                  # parameters ROSE is allowed to vary during MCMC
      thickness: [10, 100]    # [min, max] in Å
      rho: [1.0, 5.0]        # [min, max] in 10⁻⁶ Å⁻²
      interface: [1, 15]

  - name: layer_B
    rho: 4.5
    thickness: 50         # controllable — this is what you optimise over
    interface: 5

  - name: Si              # substrate — always last
    rho: 2.07

# ── Instrument / experiment settings ─────────────────────
experiment:
  q_min: 0.008            # Minimum Q (Å⁻¹)
  q_max: 0.2              # Maximum Q (Å⁻¹)
  q_points: 50            # Number of log-spaced Q points (5–1000)
  dq_over_q: 0.025        # Fractional Q resolution dQ/Q
  relative_error: 0.10    # Relative reflectivity uncertainty dR/R
  step_interfaces: false   # refl1d step-interface mode
  data_file: null          # Path to 4-column measurement file (optional)

# ── Optimisation settings ────────────────────────────────
optimization:
  param: layer_B thickness                # parameter to sweep
  param_values: [20, 30, 40, 50, 60, 70, 80]  # candidate values (Å)
  parameters_of_interest: [layer_A thickness]  # marginal entropy focus
  num_realizations: 5                     # noise realisations (1–100)
  mcmc_steps: 3000                        # MCMC steps after burn-in
  entropy_method: kdn                     # mvn or kdn
```

### Layer fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | **yes** | Layer identifier — used in `optimization.param` and `optimization.parameters_of_interest` |
| `rho` | no (default 0) | Real SLD in 10⁻⁶ Å⁻² |
| `irho` | no (default 0) | Imaginary SLD |
| `thickness` | no (default 0) | Layer thickness in Å |
| `interface` | no (default 0) | Interfacial roughness in Å |
| `fit` | no | Dict of `{param: [min, max]}` bounds for MCMC fitting |

The `fit` block determines which parameters are *variable* (fitted by MCMC)
vs. *fixed*.  Only `thickness`, `interface`, `rho`, and `irho` are
supported fit keys.

For variable parameters, the **fit range** `[min, max]` defines:
- The **uniform prior** for information gain — wider ranges mean more
  prior uncertainty.
- The **MCMC bounds** — the sampler explores only within this region.
- The **truth sampling range** — for each noise realization, ROSE draws
  a random "true" value uniformly from the fit range to generate the
  synthetic data. This ensures the information gain is averaged over
  all possible truths, not biased by the initial value.

The initial values (`rho`, `thickness`, etc.) are used only as the
**MCMC starting point**. They do not affect the information gain
calculation.

You must have at least two layers (incident medium + substrate).

### Experiment fields

| Field | Default | Description |
|-------|---------|-------------|
| `q_min` | 0.008 | Minimum Q in Å⁻¹ |
| `q_max` | 0.2 | Maximum Q in Å⁻¹ |
| `q_points` | 50 | Number of log-spaced Q points (5–1000) |
| `dq_over_q` | 0.025 | Fractional Q resolution dQ/Q |
| `relative_error` | 0.10 | Relative reflectivity uncertainty dR/R |
| `step_interfaces` | none | Enable refl1d step-interface mode |
| `data_file` | none | Path to a 4-column measurement file (Q, R, dR, dQ) |

When `data_file` is set (or `--data-file` is passed on the CLI), the
Q grid and resolution come from the measurement instead of the
`q_min`/`q_max`/`q_points` settings.  The CLI `--data-file` flag
overrides the YAML value.

### Optimization fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `param` | **yes** | — | Name of the parameter to sweep (e.g. `layer_B thickness`) |
| `param_values` | **yes** | — | List of candidate values to evaluate |
| `parameters_of_interest` | no | all variable | Restrict info-gain to these parameters |
| `num_realizations` | no | 3 | Noise realisations per candidate value (1–100) |
| `mcmc_steps` | no | 2000 | MCMC steps after burn in (100–100,000) |
| `entropy_method` | no | `kdn` | `mvn` (multivariate normal) or `kdn` (kernel density) |

### JSON alternative

The same model as JSON:

```json
{
  "name": "Layer A on Layer B",
  "layers": [
    {"name": "air", "rho": 0.0, "thickness": 0, "interface": 5},
    {"name": "layer_A", "rho": 3.0, "thickness": 40, "interface": 8,
     "fit": {"thickness": [10, 100], "rho": [1.0, 5.0], "interface": [1, 15]}},
    {"name": "layer_B", "rho": 4.5, "thickness": 50, "interface": 5},
    {"name": "Si", "rho": 2.07}
  ],
  "experiment": {"q_min": 0.008, "q_max": 0.2, "q_points": 50,
                  "dq_over_q": 0.025, "relative_error": 0.10},
  "optimization": {"param": "layer_B thickness",
                    "param_values": [20, 40, 60, 80],
                    "num_realizations": 5, "mcmc_steps": 3000}
}
```

---

## Step 2: Inspect parameters

Before running the optimisation, verify that ROSE reads your model
correctly:

```bash
rose inspect examples/models/layer_a_on_b.yaml
```

Output:

```
Variable (fitted) parameters:
  Name                           Value      Bounds
  ------------------------------------------------------------
  layer_A thickness              40         (10.0, 100.0)
  layer_A rho                    3          (1.0, 5.0)
  layer_A interface              8          (1.0, 15.0)
```

Add `--verbose` to also see fixed parameters:

```bash
rose inspect --verbose examples/models/layer_a_on_b.yaml
```

Check that:

- **Variable parameters** include everything you want MCMC to explore.
- **Fixed parameters** include the one you plan to optimise over
  (`layer_B thickness` in this example) — it must *not* appear in a `fit`
  block, because ROSE sets it to each candidate value externally.

---

## Step 3: Run the optimisation

Since all settings live in the YAML file, the CLI command is simple:

```bash
rose optimize examples/models/layer_a_on_b.yaml
```

This will:

1. Read `optimization.param_values` (20, 30, 40, …, 80 Å) and for each
   value set `layer_B thickness` to that value.
2. Simulate noisy reflectivity curves (`num_realizations` times).
3. Run MCMC with the configured number of steps on each.
4. Compute the information gain ΔH for the `parameters_of_interest`.
5. Print a summary table and save results + plots to `results/`.

### CLI options

The few remaining CLI options control output and runtime behaviour:

| Option | Default | Description |
|--------|---------|-------------|
| `--data-file` | *(none)* | 4-column measurement file — overrides Q-grid from YAML |
| `--output-dir` | `results` | Where to save JSON and plots |
| `--parallel / --sequential` | parallel | Use multiprocessing or run serially |
| `--verbose` | off | Debug-level logging |

Example with overrides:

```bash
rose optimize examples/models/layer_a_on_b.yaml \
    --data-file measurement.dat \
    --output-dir my_results \
    --verbose
```

### Understanding the output

```
=======================================================
OPTIMISATION RESULTS
=======================================================
     Value      ΔH (bits)        ± std
-------------------------------------------------------
    20.000         1.2345       0.1234
    40.000         2.3456       0.2345
    60.000         1.8765       0.1876
    80.000         1.1234       0.1123

Optimal value: 40.000
Max ΔH:        2.3456 ± 0.2345 bits
```

- **ΔH (bits)** — expected information gain: how much the posterior
  distribution narrows compared to the prior. Higher is better.
- **± std** — standard deviation across noise realisations. Large
  values indicate sensitivity to noise.
- **Optimal value** — the candidate that maximises average ΔH.

### Output files

All files are written to `--output-dir` (default: `results/`):

| File | Description |
|------|-------------|
| `optimization_results.json` | Full results including per-realisation SLD and reflectivity data |
| `information_gain.png` | Bar chart of ΔH vs. parameter value |
| `simulated_data_<value>.png` | Reflectivity curves for each candidate value |
| `sld_contour_<value>.png` | SLD profile contour plots |

---

## Step 4 (optional): Regenerate plots

If you want to recreate plots from a previous run without re-running the
optimisation:

```bash
rose report \
    --result-file results/optimization_results.json \
    --output-dir new_plots
```

---

## Worked example: Layer A on Layer B

**Goal**: Find the thickness of layer B that makes a measurement most
sensitive to changes in layer A's thickness.

```bash
# 1. Inspect to verify parameters
rose inspect examples/models/layer_a_on_b.yaml

# 2. Run — settings are already in the YAML file
rose optimize examples/models/layer_a_on_b.yaml

# 3. Look at the output
ls results/
cat results/optimization_results.json | python3 -m json.tool | head -30
```

The `parameters_of_interest: [layer_A thickness]` setting in the YAML
tells ROSE to compute the entropy only over the marginal posterior for
layer A's thickness, ignoring the other variable parameters. This focuses
the information gain metric on exactly the quantity you want to measure.

---

## Writing your own model

1. Copy `examples/models/layer_a_on_b.yaml` as a starting point.
2. Edit the `layers` list to match your sample stack.
3. Add `fit` blocks to layers with uncertain parameters.
4. Leave the controllable parameter **without** a `fit` block.
5. Add an `experiment` section to match your instrument's Q range and
   resolution.
6. Add an `optimization` section specifying what to optimise.
7. Run `rose inspect your_model.yaml` to check.
8. Run `rose optimize your_model.yaml` to find the optimal value.

### Tips

- The first layer should be the incident medium (air, D₂O, etc.).
- The last layer is the substrate (Si, SiO₂, sapphire, etc.).
- SLD values (`rho`) are in units of 10⁻⁶ Å⁻².
- If you have real measurement data, pass `--data-file` to use its
  Q-grid and resolution instead of the YAML experiment settings.
- Start with fewer realisations and MCMC steps for a quick check,
  then increase for production runs.
- Use `entropy_method: mvn` for speed or `kdn` for accuracy with
  non-Gaussian posteriors.
- Set `dq_over_q` to match your instrument's resolution (e.g. `0.02`
  for 2% dQ/Q) and `relative_error` for the expected counting-statistics
  noise level.
