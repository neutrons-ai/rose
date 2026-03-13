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
| `alternate_models` | no | `[]` | List of alternate model hypotheses (see below) |
| `discrimination_method` | no | `bic` | `bic` or `evidence` (harmonic mean estimator) |
| `discrimination_mode` | no | `report` | `report` (side-by-side) or `penalize` (effective ΔH) |
| `alt_mcmc_steps` | no | same as `mcmc_steps` | Separate MCMC step count for alternate model fits |

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

| Option | Default | Description |
|--------|---------|-------------|
| `--data-file` | *(none)* | 4-column measurement file — overrides Q-grid from YAML |
| `--output-dir` | `results` | Where to save JSON and plots |
| `--parallel / --sequential` | parallel | Use multiprocessing or run serially |
| `--workers N` | auto | Max parallel workers (default: min(tasks, CPUs, 8)) |
| `--save-problems` | off | Export refl1d FitProblem JSON files via `bumps.serialize` |
| `--verbose` | off | Debug-level logging |

Example with overrides:

```bash
rose optimize examples/models/layer_a_on_b.yaml \
    --data-file measurement.dat \
    --output-dir my_results \
    --verbose
```

### Understanding the output

With no model discrimination:

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

With model discrimination in **penalize** mode:

```
=======================================================
OPTIMISATION RESULTS
=======================================================
     Value      ΔH (bits)        ± std   P(primary)     Eff. ΔH
---------------------------------------------------------------------------
     0.000         3.1234       0.3100        0.520     1.6242
     3.000         2.5678       0.2500        0.890     2.2853
     6.000         1.8765       0.1800        0.970     1.8202

Optimal value (penalized): 3.000
Effective ΔH:  2.2853 bits
Raw ΔH:        2.5678 ± 0.2500 bits
```

- **ΔH (bits)** — expected information gain: how much the posterior
  distribution narrows compared to the prior.  Higher is better.
- **± std** — standard deviation across noise realisations.  Large
  values indicate sensitivity to noise.
- **P(primary)** — average probability that the data support the
  primary model over all alternates (only with `alternate_models`).
- **Eff. ΔH** — penalized information gain: ΔH × P(primary)
  (only in `penalize` mode).
- **Optimal value** — the candidate that maximises average ΔH
  (or effective ΔH in penalize mode).

### Output files

All files are written to `--output-dir` (default: `results/`):

| File | Description |
|------|-------------|
| `optimization_results.json` | Full results including per-realisation SLD, reflectivity, and discrimination data |
| `information_gain.png` | ΔH vs. parameter value (with penalized ΔH overlay in penalize mode) |
| `simulated_data_<i>.png` | Reflectivity curves for each candidate value |
| `sld_contours_<i>.png` | SLD profile contour plots with 90% confidence bands |
| `model_discrimination.png` | P(primary) + ΔH twin-axis plot (if alternates defined) |
| `problems/*.json` | Serialised refl1d FitProblem files (with `--save-problems`) |

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

---

## Model discrimination

Information gain alone can be misleading — a measurement condition may
yield high ΔH for the primary model but fail to distinguish it from
a simpler alternative.  Model discrimination addresses this.

### Defining alternate models

Alternate models are defined inline in the `optimization` section as
a list of modifications applied to the primary layer stack:

```yaml
optimization:
  param: THF rho
  param_values: [0, 1, 2, 3, 4, 5, 6, 7]
  parameters_of_interest: [CuOx rho, CuOx thickness]
  num_realizations: 25
  mcmc_steps: 5000

  discrimination_method: bic
  discrimination_mode: penalize

  alternate_models:
    - name: no_oxide
      modifications:
        - action: remove
          layer: CuOx
        - action: modify
          layer: Cu
          fit:
            thickness: [500, 600]
        - action: modify
          layer: THF
          fit:
            interface: [15, 100]
```

### Modification actions

| Action | Required fields | Description |
|--------|----------------|-------------|
| `remove` | `layer` | Remove a layer from the stack |
| `modify` | `layer`, optional `set` and `fit` | Change property values or fit ranges on an existing layer |
| `add` | `layer` (dict), `after` or `before` | Insert a new layer at a specified position |

The `set` dict changes fixed values (e.g. `set: {interface: 15}`).
The `fit` dict adds or changes fit ranges (e.g. `fit: {interface: [3, 40]}`).

### Discrimination methods

- **BIC** (`bic`): Bayesian Information Criterion.  Computes
  ΔBIC = BIC(alternate) − BIC(primary); positive values favour the
  primary.  Fast — no additional MCMC runs beyond those already done.
- **Evidence** (`evidence`): Harmonic mean estimator of the marginal
  likelihood from the DREAM chains.  Computes log Bayes factor.
  Higher variance but doesn't assume model nesting.

Both produce $P(\text{primary} | \text{data})$ via a logistic transform.

### Discrimination modes

- **report**: ΔH and P(primary) are displayed side-by-side.  The optimal
  value is still chosen by raw ΔH.  Use this when you want to see both
  metrics but make your own judgement.
- **penalize**: Computes an effective information gain:
  $\Delta H_{\text{eff}} = \Delta H \times \bar{P}(\text{primary})$.
  Conditions where the primary can't be distinguished from alternatives
  get their gain reduced.  The optimal value is chosen by max effective ΔH.

### Output with discrimination

When alternates are configured, the results JSON contains a
`"discrimination"` key with per-value metrics.  The report generates
a `model_discrimination.png` plot showing P(primary) on the left axis
and information gain on the right axis.  In penalize mode, the
`information_gain.png` plot overlays both raw and penalized ΔH curves.

### Saving FitProblem files

Pass `--save-problems` to export the refl1d `FitProblem` objects used
during MCMC as JSON files (via `bumps.serialize`).  These capture the
exact model, data, and parameter state used for fitting:

```bash
rose optimize examples/models/cu_thf.yaml --save-problems
```

Files are saved to `{output-dir}/problems/`:
- `step_0_value_0.0_primary.json` — primary model
- `step_0_value_0.0_no_oxide.json` — alternate model(s)

Reload with:

```python
from bumps.serialize import load_file
problem = load_file("results/problems/step_0_value_0.0_primary.json")
```
