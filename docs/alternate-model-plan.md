# Plan: Alternate Model Discrimination in Experiment Optimization

## TL;DR

The current optimization maximizes information gain about parameters within a **single model**. But high ΔH doesn't guarantee the experiment can distinguish the intended model from simpler alternatives (e.g., CuOx layer vs. larger Cu roughness). We add a **model discrimination** dimension: users define alternate hypotheses inline in the YAML, both primary and alternates are fit per realization, and two selectable discrimination metrics (BIC, log-evidence via harmonic mean) are computed and reported alongside parameter information gain.

---

## Context

In the `cu_thf.yaml` example:
- **Primary model**: THF | CuOx | Cu | Ti | Si
- **Alternate model**: THF | Cu (with larger roughness/thickness) | Ti | Si — no CuOx

At **low THF SLD** (~0): High ΔH but CuOx is indistinguishable from Cu roughness (alternate fits equally well).
At **high THF SLD** (~6-7): High ΔH AND CuOx creates contrast → alternate fits poorly → discriminating.

Current ROSE treats both cases identically. This plan adds the machinery to detect the difference.

---

## User choices
- **Discrimination methods**: Both BIC and log-evidence (via harmonic mean of logp from DREAM chains) as selectable methods (`discrimination_method: bic | evidence`)
- **Alternate model format**: Inline modifications in the primary YAML (diff-based: remove/add/modify layers)
- **Scoring modes**: Default shows both metrics side by side; optional `discrimination_mode: "penalize"` multiplies info_gain by P(primary | data)
- **Alternate fitting**: Full MCMC for alternates (same as primary)

---

## Steps

### Phase 1: YAML Schema for Inline Alternate Models (3 steps)

1. **Define inline alternate model schema** in the `optimization` section of the YAML.
   Each alternate model is specified as a set of modifications to the primary layer stack:
   ```yaml
   optimization:
     alternate_models:
       - name: no_oxide
         description: "Cu roughness without CuOx layer"
         modifications:
           - action: remove
             layer: CuOx
           - action: modify
             layer: Cu
             set:
               interface: 20
             fit:
               interface: [5, 40]
               thickness: [480, 560]
     discrimination_method: bic    # or "evidence"
     discrimination_mode: report   # or "penalize"
   ```
   Supported actions: `remove` (delete a layer), `modify` (change values/fit ranges on a layer), `add` (insert a new layer at a position).
   Add these to `OPTIMIZATION_DEFAULTS` in `model_loader.py`.

2. **Validate alternate model modifications** — Add `_validate_alternate_models()` in `model_loader.py`. Check: referenced layer names exist in primary, `set` keys are valid layer properties, `fit` ranges are valid bounds, `action` is one of `remove`/`modify`/`add`. For `add`, require `after: layer_name` or `before: layer_name` plus full layer spec.

3. **Build alternate experiments from modifications** — Add `build_alternate_experiment()` in `model_loader.py`:
   - Deep-copy the primary model description dict
   - Apply each modification (remove layers, modify values/fit ranges, add layers)
   - Call existing `build_experiment()` on the modified description
   - Return a list of `(name, Experiment)` tuples
   - Primary experiment's Q-range/probe settings are reused (alternates inherit the same probe).

### Phase 2: Discrimination Engine (4 steps)

4. **Add BIC computation** — New function `compute_bic()` in `experiment_design.py`:
   - Input: bumps `FitProblem`, DREAM `state`
   - Extract best-fit logp from `state.best()` → `best_params, best_logp`
   - `k` = number of free parameters in the problem
   - `n` = number of data points (length of Q array)
   - `BIC = -2 * best_logp + k * ln(n)`
   - Return BIC value

5. **Add log-evidence via harmonic mean** — New function `compute_log_evidence()` in `experiment_design.py`:
   - Input: DREAM `state`, `portion=0.3`
   - Extract `(points, logp)` from `state.sample(portion=portion)`
   - Harmonic mean estimator: `log Z ≈ -log(mean(exp(-logp)))` (use log-sum-exp for numerical stability)
   - Known to have high variance, but cheap and uses existing chains
   - Return `log_evidence` value (in nats, convert to bits for consistency)

6. **Add `ModelDiscriminator` class** in new file `src/rose/planner/model_discriminator.py`:
   - Constructor: takes primary experiment, list of `(name, Experiment)` alternate experiments
   - `fit_alternate(alt_experiment, q_values, noisy_reflectivity, errors, dq_values, mcmc_steps)` → runs `perform_mcmc()` on alternate, returns DREAM state
   - `compute_discrimination(primary_state, alt_state, method="bic"|"evidence")`:
     - If `bic`: compute ΔBIC = BIC_alt - BIC_primary (positive favors primary)
     - If `evidence`: compute Bayes factor = exp(log_Z_primary - log_Z_alt) and log(BF)
   - `model_probability(delta_metric, method)` → P(primary | data) approximation:
     - BIC: `P = 1 / (1 + exp(-ΔBIC/2))`
     - Evidence: `P = BF / (1 + BF)`

7. **Combined scoring function** — In `model_discriminator.py`:
   - `combine_scores(info_gain, model_probabilities, mode="report"|"penalize")`
   - `"report"`: return both as-is (no combination)
   - `"penalize"`: `effective_info_gain = info_gain × mean(P(primary | data))` across all alternates

### Phase 3: Optimizer Integration (3 steps)

8. **Thread alternate models through optimizer** — In `optimizer.py`:
   - `_evaluate_single_realization()` gains an optional `alternate_experiments` parameter (list of `(name, sample_stack)` tuples) and `discrimination_method` parameter
   - After fitting primary model, for each alternate: create fresh `FitProblem`, call `perform_mcmc()` with same noisy data, compute discrimination metric
   - Return expanded tuple including discrimination dict: `{alt_name: {"delta_metric": float, "model_prob": float}}`
   - `evaluate_param()` aggregates discrimination across realizations (mean, std)

9. **Update `optimize()` and `optimize_parallel()`** — Accept `alternate_experiments` and `discrimination_method` parameters, pass them through to evaluation functions.  The CLI constructs alternate experiments from the YAML before calling optimize.

10. **Update CLI** — In `cli.py`, after loading the primary model description:
    - If `desc["optimization"]` has `alternate_models`, call `build_alternate_experiment()` to produce alternate experiments
    - Pass alternates to `optimize()` / `optimize_parallel()`
    - Read `discrimination_method` and `discrimination_mode` from the YAML
    - Include discrimination data in the output JSON

### Phase 4: Results & Reporting (3 steps)

11. **Extend result types** — In `core/types.py`:
    - `RealizationData`: add `discrimination: dict[str, float]` (alt_name → Δmetric), default `{}`
    - `ParameterResult`: add `mean_discrimination: dict[str, float]`, `model_probability: dict[str, float]`
    - `OptimizationResult`: add `alternate_models: list[str]`, `discrimination_method: str`, `discrimination_mode: str`

12. **Extend result JSON** — In the CLI optimize command, include discrimination data. Backward-compatible: if no alternates, identical format to current output.

13. **Add discrimination plots** — In `report.py`:
    - New `model_discrimination.png`: dual-axis plot — left: ΔH (bits) vs param value, right: ΔBIC or log(BF) vs param value. Highlight where discrimination is strong/weak.
    - If `discrimination_mode: "penalize"`, overlay `effective_info_gain` on the info_gain plot
    - Add text annotation on info_gain.png noting which peak has better model discrimination

### Phase 5: Testing (3 steps)

14. **Unit tests for discrimination** — In new `tests/test_model_discriminator.py`:
    - Test `compute_bic()` with known logp, k, n values
    - Test `compute_log_evidence()` with synthetic logp arrays where evidence is analytically known
    - Test `model_probability()` for edge cases (ΔBIC=0, large positive, large negative)
    - Test `combine_scores()` for both modes

15. **Unit tests for YAML alternate models** — In `tests/test_model_loader.py`:
    - Test `_validate_alternate_models()`: valid modifications, invalid layer names, missing actions, add without position
    - Test `build_alternate_experiment()`: verify layer removal, modification, addition produce correct refl1d experiments

16. **Integration test** — In `tests/test_optimizer.py` (or separate slow test):
    - Construct a simple 3-layer vs 2-layer scenario with known discrimination properties
    - Verify discrimination metric is higher when models are truly different
    - Verify backward compatibility: optimization without alternates produces unchanged results

### Phase 6: Documentation (1 step)

17. **Update docs and examples**:
    - `docs/ground_truths.md`: Document the discrimination algorithm, BIC and evidence formulas, design decisions
    - `docs/use-case-1.md`: Add alternate model workflow section
    - `examples/models/cu_thf.yaml`: Add `alternate_models` section with the no-oxide modification inline
    - `docs/use-case-3.md` (new): "Model discrimination — choosing experiments that distinguish competing hypotheses"

---

## Relevant Files

- `src/rose/planner/model_discriminator.py` — **New**: `ModelDiscriminator`, `compute_bic()`, `compute_log_evidence()`, `combine_scores()`
- `src/rose/planner/optimizer.py` — Modify `_evaluate_single_realization()`, `evaluate_param()`, `optimize()`, `optimize_parallel()` to accept and process alternates
- `src/rose/planner/experiment_design.py` — Reference for `ExperimentDesigner` pattern; BIC/evidence helpers could also live here
- `src/rose/planner/mcmc_sampler.py` — `perform_mcmc()` reused for alternate fitting (no changes needed)
- `src/rose/planner/model_loader.py` — `_validate_alternate_models()`, `build_alternate_experiment()`, update `_validate_optimization()`
- `src/rose/core/types.py` — Add discrimination fields to dataclasses
- `src/rose/planner/report.py` — New discrimination plot
- `src/rose/cli.py` — Load alternates from YAML, pass to optimizer, write extended JSON
- `examples/models/cu_thf.yaml` — Add inline alternate_models modifications
- `docs/ground_truths.md`, `docs/use-case-1.md`, `docs/use-case-3.md` — Documentation updates
- `tests/test_model_discriminator.py` — **New**: discrimination unit tests
- `tests/test_model_loader.py` — New tests for alternate model validation/building

---

## Verification

1. **Unit**: `compute_bic()` returns expected value for known inputs (logp=-100, k=3, n=60 → BIC = 200 + 3·ln(60))
2. **Unit**: `compute_log_evidence()` with uniform logp=-50 array → log_Z ≈ 50 (sanity check)
3. **Unit**: `model_probability(ΔBIC=10)` ≈ 0.993, `model_probability(ΔBIC=0)` = 0.5
4. **Unit**: `combine_scores(info_gain=3.0, probs=[0.99], mode="penalize")` ≈ 2.97
5. **Unit**: YAML validation rejects alternate with nonexistent layer name
6. **Unit**: `build_alternate_experiment()` removes CuOx layer → alternate has one fewer layer
7. **Integration**: cu_thf with no-oxide alternate — high THF SLD has higher combined score
8. **Backward compat**: layer_a_on_b.yaml without alternates → identical JSON output
9. **Regression**: All 83 existing tests pass
10. **Visual**: Report plots for cu_thf show discrimination curve alongside info_gain

---

## Decisions

- **Inline modifications** (not external files): alternate models defined as diffs to the primary YAML (`remove`, `modify`, `add` actions). Keeps everything in one file and avoids managing multiple YAML files.
- **Two discrimination methods**: BIC (fast, interpretable) and harmonic-mean log-evidence (uses existing DREAM chains). Both available from v1 via `discrimination_method: bic | evidence`.
- **Full MCMC for alternates**: Same `mcmc_steps` as primary by default. Optional `alt_mcmc_steps` override for performance tuning.
- **New file for discriminator**: `model_discriminator.py` keeps discrimination logic separate from `experiment_design.py` (which handles entropy).
- **Harmonic mean, not thermodynamic integration**: True thermodynamic integration requires running MCMC at multiple temperatures (simulated tempering). bumps DREAM doesn't expose this. The harmonic mean estimator works on existing chains. It has known high-variance issues but is a reasonable first approximation. Can be upgraded to stepping-stone sampling later if needed.
- **Scope included**: BIC, harmonic-mean evidence, inline YAML alternates, combined scoring, reporting.
- **Scope excluded**: True thermodynamic integration, automatic LLM generation of alternate hypotheses, model averaging, nested sampling.
