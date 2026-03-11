# Use-Case 2: Text-to-Model Planning with an LLM

This guide walks you through use-case 2 — describing a sample and
hypothesis in plain language (a text file), letting an LLM generate
a ROSE model, and then running the Bayesian optimisation.

## Prerequisites

```bash
# Install ROSE with the LLM extras
pip install -e ".[dev,llm]"

# Verify the new commands are available
rose plan --help
rose plan-and-optimize --help
```

You also need an LLM API key. Copy the env template and fill in your key:

```bash
cp .env.example .env
# Edit .env and set LLM_API_KEY=sk-...
```

ROSE reads the `OPENAI_API_KEY` environment variable (or `LLM_API_KEY`)
automatically via LangChain. You can also pass `--model-name` on the
CLI to select a different model.

## Overview

The workflow has three paths, depending on how much control you want:

```
                         ┌────────────────┐
     ┌──────────┐       │  rose plan     │       ┌──────────┐
     │  Write   │ ───▶  │  (LLM → YAML)  │ ───▶  │  Review  │
     │  .txt    │       └────────────────┘       │  model   │
     └──────────┘               │                └────┬─────┘
                                │                     │
                                ▼                     ▼
                    ┌────────────────────┐    ┌──────────────┐
                    │ rose plan-and-     │    │ rose optimize│
                    │ optimize (1 step)  │    │ (use-case 1) │
                    └────────────────────┘    └──────────────┘
```

- **Path A** — `rose plan` → review → `rose optimize` (maximum control)
- **Path B** — `rose plan-and-optimize` (single command, end-to-end)

---

## Step 1: Write a description file

Write a plain text file describing your sample, what you want to study,
and any constraints. The LLM infers everything: layer stack, SLD values,
fit ranges, optimisation target, candidate values, and instrument settings.

### Example: `polymer_on_gold.txt`

```text
Polystyrene thin film (~50 nm) on a gold adhesion layer deposited on
a silicon substrate, measured in air.  The polymer film thickness and
SLD are uncertain.

Find the optimal gold adhesion layer thickness to maximise sensitivity
to changes in the polymer film thickness.  Evaluate gold thicknesses
from 50 to 300 Å in steps of 50 Å.
```

### Example: `protein_adsorption.txt`

```text
Protein adsorption experiment at a silicon/water interface.
A thin SiO2 native oxide (~15 Å) sits on the silicon substrate.
The protein adsorbs on top (estimated ~30 Å, but very uncertain),
and the whole system is immersed in deuterated water (D2O).
Both the protein layer thickness and SLD are unknown.

Determine which protein layer thickness best reveals the protein
SLD and coverage.  Sweep protein thicknesses from 10 to 80 Å.
```

### Tips for writing good descriptions

- **Name the materials** — use common names ("gold", "silicon",
  "polystyrene") or formulas ("Au", "SiO₂", "D₂O").
- **Mention thicknesses** when you know them (even approximate).
- **State what's uncertain** — the LLM will add fit ranges for these.
- **State your scientific goal** — what do you want to optimise?
- **Mention candidate values** if you have preferences.
- **Specify instrument settings** only if they differ from defaults
  (Q range 0.008–0.2 Å⁻¹, 50 points, 2.5% resolution).

You can also use a YAML file with a `description` key:

```yaml
description: >
  Polystyrene thin film (~50 nm) on gold on silicon, measured in air.
  Find optimal gold thickness for polymer sensitivity.
```

---

## Step 2: Generate the model

### Path A: Generate, review, then optimise

```bash
# Generate a YAML model and save it
rose plan examples/queries/polymer_on_gold.txt \
    --output generated_model.yaml

# Review what the LLM produced
cat generated_model.yaml

# If it looks good, run the optimisation (use-case 1 workflow)
rose optimize generated_model.yaml --output-dir results/
```

The generated file is a standard ROSE YAML model (identical format to
hand-authored models in use-case 1). You can edit it before optimising.

### Path B: One-step end-to-end

```bash
rose plan-and-optimize examples/queries/polymer_on_gold.txt \
    --output-dir results/
```

This runs the LLM generation and optimisation in sequence, saving the
generated model as `results/generated_model.yaml`.

### CLI options for `rose plan`

| Option | Default | Description |
|--------|---------|-------------|
| `--output` / `-o` | *(stdout)* | Save generated YAML to a file |
| `--model-name` | `gpt-4o` | LLM model to use |
| `--temperature` | `0.2` | Sampling temperature (lower = more deterministic) |
| `--verbose` | off | Debug logging |

### CLI options for `rose plan-and-optimize`

Includes all `plan` options plus:

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `results` | Where to save model + optimisation output |
| `--data-file` | none | Optional measured data file (4-column Q, dQ, R, dR) |
| `--parallel / --sequential` | parallel | Run optimiser in parallel |
| `--workers` | auto | Max parallel workers |
| `--verbose` | off | Debug logging |

---

## Step 3: Review the generated model

The LLM generates a YAML model that looks like this:

```yaml
description: >
  Polystyrene thin film on a gold adhesion layer on silicon,
  optimising gold thickness for polymer sensitivity.

layers:
  - name: air
    rho: 0.0
    thickness: 0
    interface: 0
  - name: polymer
    rho: 1.412
    thickness: 50
    interface: 5
    fit:
      thickness: [10, 200]
      interface: [1, 15]
  - name: gold_layer
    rho: 4.662
    thickness: 100
    interface: 3
  - name: Si
    rho: 2.074
experiment:
  q_min: 0.008
  q_max: 0.25
  q_points: 60
  dq_over_q: 0.025
  relative_error: 0.05
optimization:
  param: gold_layer thickness
  param_values: [50, 100, 150, 200, 250, 300]
  parameters_of_interest: [polymer thickness]
  num_realizations: 5
  mcmc_steps: 3000
  entropy_method: kdn
```

Things to verify:

- **SLD values** — Are they reasonable for the materials?
  (`Au` ≈ 4.66, `Si` ≈ 2.07, polystyrene ≈ 1.41)
- **Layer order** — Top (air) to bottom (substrate)
- **Fit ranges** — Are the `[min, max]` bounds physically sensible?
- **Optimization param** — Does it match your scientific question?
- **Param values** — Is the sweep range reasonable?

Use `rose inspect` to double-check:

```bash
rose inspect generated_model.yaml
```

---

## How the LLM pipeline works

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Plain text  │───▶│  SLD table   │───▶│  LLM call    │───▶│  Validator   │
│  description │    │  in system   │    │  (ChatOpenAI)│    │  (schema     │
│  (.txt)      │    │  prompt      │    │              │    │   check)     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
                                                            valid? │
                                                           ┌───────┴──────┐
                                                     yes ──▶ Return YAML  │
                                                           │              │
                                                     no  ──▶ Retry with   │
                                                           │ error msg    │
                                                           └──────────────┘
```

1. **Load description** — The text file is read as-is. No structured
   schema or Pydantic validation on the input — just a plain string.
2. **Prompt construction** — A system prompt teaches the LLM the ROSE YAML
   schema (layer format, experiment defaults, optimization structure) and
   includes an SLD reference table (28+ materials). The user prompt
   contains the plain-text description.
3. **LLM generation** — The model (default: `gpt-4o`, temperature 0.2)
   generates a complete YAML model file with `description`, `layers`,
   `experiment`, and `optimization` sections.
4. **Validation** — The output is checked against the ROSE schema
   (required keys, layer names, fit ranges, param references). If invalid,
   the errors are fed back and the LLM retries (up to 2 retries).

**Security**: The LLM generates YAML data, never executable code. All
output is parsed with `yaml.safe_load` and validated against a strict
schema before use.

---

## Supported materials

ROSE includes a database of common reflectometry materials. You can
mention any of these in your description and the LLM will use the
correct SLD values:

| Name | Formula | SLD (10⁻⁶ Å⁻²) |
|------|---------|-----------------|
| silicon | Si | 2.07 |
| gold | Au | 4.66 |
| copper | Cu | 6.55 |
| heavy water | D₂O | 6.37 |
| water | H₂O | −0.56 |
| polystyrene | C₈H₈ | 1.41 |
| silicon dioxide | SiO₂ | 3.47 |
| air / vacuum | — | 0.0 |

See the full list with:
```python
from rose.modeler.sld_database import list_materials
for m in list_materials():
    print(f"{m.name:10s}  SLD={m.sld:.3f}")
```

---

## Worked example: Polymer on gold

**Goal**: Find the gold layer thickness that makes a reflectometry
measurement most sensitive to the polymer film thickness.

```bash
# 1. Set up the API key
cp .env.example .env
# Edit .env: set LLM_API_KEY=sk-...

# 2. Generate the model
rose plan examples/queries/polymer_on_gold.txt \
    --output polymer_model.yaml

# 3. Review it
rose inspect polymer_model.yaml

# 4. Run the optimisation
rose optimize polymer_model.yaml --output-dir polymer_results/

# 5. View results
ls polymer_results/
```

Or do it all in one step:

```bash
rose plan-and-optimize examples/queries/polymer_on_gold.txt \
    --output-dir polymer_results/
```

---

## Worked example: Protein adsorption

**Goal**: Determine the protein layer thickness at a solid/liquid interface.

```bash
rose plan examples/queries/protein_adsorption.txt \
    --output protein_model.yaml

cat protein_model.yaml

rose optimize protein_model.yaml --output-dir protein_results/
```

---

## Tips

- **Review the generated model** before running expensive optimisations.
  The LLM may pick different initial values or fit ranges than you expect.
- **Lower temperature** (0.1–0.2) produces more consistent YAML. Higher
  temperatures (0.5+) may be more creative but risk invalid output.
- **Be specific** about materials and thicknesses in your description —
  the more detail you give, the better the LLM's output.
- **Start small** — adjust `num_realizations` and `mcmc_steps` in the
  generated YAML before running expensive optimisations.
- **Edit the output** — the generated model is a plain YAML file. Feel
  free to tweak it before optimising.
- **Local models** — set `LLM_BASE_URL` in `.env` to use a local
  Ollama/vLLM instance. Larger models (70B+) tend to produce better YAML.

## Differences from use-case 1

| | Use-case 1 | Use-case 2 |
|--|-----------|-----------|
| **Model authoring** | Write YAML by hand | LLM generates YAML from plain text |
| **Input** | YAML model file | Text file (.txt) describing the sample |
| **SLD values** | You look them up | Database + LLM handle it |
| **Fit ranges** | You specify `fit:` blocks | LLM infers from your description |
| **Commands** | `rose optimize` | `rose plan` → `rose optimize` |
| **Requirements** | refl1d only | refl1d + LLM API key |
| **Output** | Same JSON + plots | Same JSON + plots |
