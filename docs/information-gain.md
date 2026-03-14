# Information Gain Algorithm

This document describes the Bayesian experiment design algorithm implemented in
ROSE. The approach follows **Treece et al., _J. Appl. Cryst._ (2019), 52, 47–59**
and uses Shannon information gain to identify optimal experimental conditions for
neutron reflectometry.

---

## 1. Overview

The central question ROSE answers is:

> Given a model of a thin-film sample, **which experimental condition**
> (e.g. solvent contrast, measurement angle, temperature) **maximises what we
> learn** about the parameters we care about?

"What we learn" is quantified as the expected **Shannon information gain** —
the reduction in uncertainty (entropy) about model parameters when we go from
our prior beliefs to the posterior distribution informed by data:

$$
\Delta H = H_{\text{prior}} - \mathbb{E}\!\left[H_{\text{posterior}}\right]
$$

The candidate condition that yields the largest $\Delta H$ is the most
informative experiment.

---

## 2. Mathematical Framework

### 2.1 Prior Entropy

All fitted parameters are assumed to have **uniform priors** over their declared
bounds $[a_i, b_i]$. The entropy of a uniform distribution on $[a, b]$ is
$\log_2(b - a)$ bits.

For $d$ parameters of interest the total prior entropy is:

$$
H_{\text{prior}} = \sum_{i=1}^{d} \log_2(b_i - a_i)
$$

This is computed once at the start of an optimization run because the prior
does not depend on the experimental condition being tested.

### 2.2 Posterior Entropy

After observing (simulated) data, the prior collapses to a posterior
distribution. The posterior entropy is estimated from MCMC samples using one of
two methods:

#### Multivariate Normal (MVN)

Fit a multivariate Gaussian to the MCMC samples and use the analytic entropy
formula:

$$
H_{\text{MVN}} = \frac{1}{2} \ln\!\left[(2\pi e)^d \det(\Sigma)\right] \;\Big/\; \ln 2
$$

where $\Sigma$ is the sample covariance matrix. This is fast and stable but
assumes the posterior is approximately Gaussian — a good approximation when
the MCMC chain has converged to a single, well-defined mode.

If the covariance matrix is singular (degenerate posterior), a small ridge
regularisation $\Sigma \leftarrow \Sigma + 10^{-10} I$ is applied.

#### Kernel Density Estimation (KDN)

Estimate the posterior density non-parametrically with `scipy.stats.gaussian_kde`
and compute the Monte Carlo entropy:

$$
H_{\text{KDE}} = -\frac{1}{N}\sum_{j=1}^{N} \log_2 \hat{p}(\mathbf{x}_j)
$$

where $\hat{p}$ is the kernel density estimate evaluated at each MCMC sample
$\mathbf{x}_j$. This captures multi-modal or skewed posteriors that the MVN
method would miss, at the cost of higher variance and computation time. If
KDE fails (e.g. too few samples), the code falls back to MVN automatically.

**Default:** KDN is the default method because reflectometry posteriors are
often non-Gaussian.

### 2.3 Information Gain

For a single candidate condition $c$ and a single noise realization $k$:

$$
\Delta H_k(c) = H_{\text{prior}} - H_{\text{posterior}}^{(k)}(c)
$$

Multiple noise realizations are averaged to obtain the **expected** information
gain for that condition:

$$
\overline{\Delta H}(c) = \frac{1}{K}\sum_{k=1}^{K} \Delta H_k(c)
$$

The optimal condition is $c^* = \arg\max_c \;\overline{\Delta H}(c)$.

### 2.4 Parameters of Interest (Marginalisation)

Users can specify a subset of parameters as "parameters of interest." When
this is set, only those columns of the MCMC sample array are used for entropy
calculations. This marginalises out nuisance parameters and focuses the
information gain on the quantities the experimenter actually wants to
constrain.

---

## 3. Algorithm Pipeline

The full optimization workflow for each candidate parameter value proceeds as
follows. Steps 3a–3f are repeated $K$ times (noise realizations) and the
resulting information gains are averaged.

```
┌─────────────────────────────────────────────────────┐
│  1. Compute H_prior from parameter bounds (once)    │
├─────────────────────────────────────────────────────┤
│  For each candidate value c in the grid:            │
│  ┌─────────────────────────────────────────────┐    │
│  │ 2. Set the controllable parameter to c      │    │
│  │    (e.g., solvent SLD = 5.8)                │    │
│  ├─────────────────────────────────────────────┤    │
│  │ For each realization k = 1..K:              │    │
│  │  3a. Draw "true" values from the prior      │    │
│  │  3b. Simulate reflectivity from truth       │    │
│  │  3c. Add instrumental noise                 │    │
│  │  3d. Restore original parameter values      │    │
│  │  3e. Run MCMC to sample the posterior       │    │
│  │  3f. Compute ΔH_k = H_prior − H_posterior  │    │
│  ├─────────────────────────────────────────────┤    │
│  │ 4. Average ΔH over realizations             │    │
│  └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│  5. Select c* = argmax of mean ΔH                   │
└─────────────────────────────────────────────────────┘
```

### Step-by-step detail

#### Step 1 — Prior Entropy

`ExperimentDesigner.prior_entropy()` sums $\log_2(b_i - a_i)$ over every
parameter of interest. This value is constant for the entire optimization.

#### Step 2 — Set Candidate Condition

`ExperimentDesigner.set_parameter_to_optimize(name, value)` sets a **fixed**
(non-fitted) parameter on the model. For example, in a contrast-variation
experiment the solvent SLD is not a fitted parameter — it is the controllable
experimental knob whose optimal setting we seek.

#### Step 3a — Draw Truth from Prior

`ExperimentDesigner.draw_truth_from_prior()` samples each **fitted** parameter
uniformly within its bounds. This sets the model to a plausible "ground truth"
so we can ask: _if these were the real parameter values, how much would we
learn?_

Drawing from the prior ensures the information gain is an expectation over all
possible truths, not conditioned on a single assumed ground truth.

#### Step 3b — Simulate Reflectivity

`experiment.reflectivity()` computes the theoretical reflectivity curve
$R(Q)$ from the refl1d model with the drawn truth values.

#### Step 3c — Add Instrumental Noise

`InstrumentSimulator.add_noise()` adds Gaussian noise:

$$
R_{\text{noisy}}(Q_i) = R(Q_i) + \mathcal{N}\!\left(0,\; \sigma_i\right),
\quad \sigma_i = \epsilon_{\text{rel},i} \cdot R(Q_i)
$$

where $\epsilon_{\text{rel},i}$ is either derived from a real data file
(inheriting the instrument's actual error profile) or set to a constant
relative error (default 10%).

The simulator can also be initialised from a measured data file so the
synthetic data inherits the real $Q$-grid, resolution $\Delta Q$, and
point-by-point error bars from an actual measurement.

#### Step 3d — Restore Starting Point

Before running MCMC the fitted parameters are reset to their original YAML
values. This prevents the MCMC sampler from starting at the (known) truth,
which would bias the posterior.

#### Step 3e — MCMC Posterior Sampling

`mcmc_sampler.perform_mcmc()` creates a fresh refl1d `FitProblem` with the
noisy data and runs the **DREAM** (DiffeRential Evolution Adaptive Metropolis)
algorithm via the `bumps` library:

1. Build a `QProbe` from $(Q, \Delta Q, R_{\text{noisy}}, \delta R)$.
2. Construct a `FitProblem` from the sample and probe.
3. Run DREAM with a configurable number of burn-in and production steps.
4. Mark outlier chains and keep the best sample.

The output is a set of MCMC samples $\{\mathbf{x}_j\}_{j=1}^{N}$ drawn from
the posterior $p(\boldsymbol{\theta} \mid D)$.

#### Step 3f — Posterior Entropy and Information Gain

1. `extract_marginal_samples()` selects only the columns corresponding to
   parameters of interest (if specified).
2. `calculate_posterior_entropy()` computes $H_{\text{posterior}}$ via MVN or
   KDN.
3. $\Delta H_k = H_{\text{prior}} - H_{\text{posterior}}$.

#### Steps 4–5 — Aggregation

The mean and standard deviation of $\Delta H$ over realizations are recorded
for each candidate value. The value with the highest mean information gain is
the recommended experimental condition.

---

## 4. Model Discrimination

Optionally, ROSE can evaluate whether the data from a given experimental
condition can **distinguish the primary model from simpler alternatives**. This
is implemented in `ModelDiscriminator`.

### 4.1 Alternate Models

Alternate hypotheses are defined in the YAML file as modifications to the
primary model (e.g. removing a layer, changing fit ranges). Each alternate
model is fitted to the same noisy data as the primary.

### 4.2 Metrics

Two discrimination metrics are available:

#### BIC (Bayesian Information Criterion)

$$
\text{BIC} = -2 \ln L_{\text{best}} + k \ln n
$$

where $L_{\text{best}}$ is the best likelihood from the MCMC chain, $k$ is
the number of free parameters, and $n$ is the number of data points. The
difference $\Delta\text{BIC} = \text{BIC}_{\text{alt}} - \text{BIC}_{\text{primary}}$
is positive when the data favours the primary model.

The approximate model probability is:

$$
P(\text{primary} \mid D) \approx \frac{1}{1 + e^{-\Delta\text{BIC}/2}}
$$

#### Log Bayes Factor (Evidence)

The log-evidence for each model is estimated using the **Newton–Raftery
harmonic mean estimator** on the existing DREAM chain:

$$
\hat{Z} = \left[\frac{1}{N}\sum_{j=1}^{N} \frac{1}{L(\mathbf{x}_j)}\right]^{-1}
$$

implemented via log-sum-exp for numerical stability. The log Bayes factor is
$\ln B = \ln Z_{\text{primary}} - \ln Z_{\text{alt}}$, and positive values
favour the primary model.

This estimator is known to have high variance but has the advantage of
requiring no additional MCMC runs beyond those already performed.

### 4.3 Combined Scoring

When `discrimination_mode: penalize` is set, the effective information gain
is:

$$
\Delta H_{\text{eff}} = \Delta H \times \bar{P}(\text{primary} \mid D)
$$

where $\bar{P}$ is the mean model probability across all alternate models.
This penalises experimental conditions that cannot disambiguate model
hypotheses: even if an experiment is highly informative for parameter
estimation, it is less useful if it cannot tell whether the model structure
itself is correct.

---

## 5. Execution Modes

### Sequential (`optimize`)

Evaluates one parameter value at a time. Each value runs $K$ noise realizations
serially. Suitable for debugging or small grids.

### Parallel (`optimize_parallel`)

Uses Python's `ProcessPoolExecutor` to distribute every `(value, realization)`
pair as an independent task. All workers stay busy even when the number of
parameter values is small. Results are collected and re-ordered to match the
original parameter grid.

Maximum parallelism is capped at `min(total_tasks, cpu_count, 8)`.

---

## 6. SLD Contour Extraction

After each MCMC fit, the code extracts **scattering length density (SLD)
depth-profile confidence bands** from the DREAM chain. This provides a
visual diagnostic: for each candidate condition, you can see how tightly the
SLD profile is constrained.

The contour extraction:

1. Draws posterior samples from the chain.
2. Computes the reflectometry profile for each sample.
3. Aligns the profiles by depth.
4. Builds a confidence band (default 90%) at each depth point.

The resulting `[z, best, low, high]` arrays are stored alongside the
information gain results for visualization.

---

## 7. Practical Example

Consider a **Cu / unknown-oxide / THF** system where we want to determine the
oxide layer's SLD and thickness. The controllable variable is the **THF solvent
SLD** (via isotopic substitution).

**Model file (`cu_thf.yaml`):**

```yaml
layers:
  - name: THF
    rho: 5.8          # solvent SLD — this is what we optimize
    thickness: 0
    interface: 16
    fit:
      interface: [15, 35]
  - name: CuOx
    rho: 5.0
    thickness: 30
    interface: 13
    fit:
      thickness: [10, 70]
      rho: [4.0, 6.0]
      interface: [10, 25]
  # ... Cu, Ti, Si substrate layers ...

optimization:
  param: THF rho                              # knob to turn
  param_values: [0, 1, 2, 3, 4, 5, 6, 7]     # candidate SLD values
  parameters_of_interest: [CuOx rho, CuOx thickness]
  num_realizations: 25
  mcmc_steps: 5000
  entropy_method: kdn
```

**What happens:**

1. $H_{\text{prior}}$ is computed from the CuOx bounds:
   $\log_2(6.0 - 4.0) + \log_2(70 - 10) \approx 1.0 + 5.9 = 6.9$ bits.
   (Interface bounds also contribute if CuOx interface is a parameter of
   interest.)
2. For each THF SLD value (0 through 7), 25 noise realizations are run.
3. Each realization draws random CuOx parameters, simulates reflectivity at
   that THF contrast, adds noise, runs DREAM MCMC, and computes
   $\Delta H$.
4. The THF SLD that produces the highest average $\Delta H$ is the recommended
   contrast for the experiment.

Additionally, with `discrimination_mode: penalize`, if an alternate model
("no oxide — just rough Cu surface") fits the data equally well at certain
contrasts, those contrasts are penalised. The recommended contrast will be one
that both constrains the oxide parameters **and** demonstrates that the oxide
layer is genuinely present.

---

## 8. Implementation Map

| Concept                      | Module                                  | Key Function / Class            |
|------------------------------|-----------------------------------------|---------------------------------|
| Prior entropy                | `planner.experiment_design`             | `ExperimentDesigner.prior_entropy()` |
| Truth sampling               | `planner.experiment_design`             | `ExperimentDesigner.draw_truth_from_prior()` |
| Posterior entropy (MVN)      | `planner.experiment_design`             | `ExperimentDesigner._posterior_entropy_mvn()` |
| Posterior entropy (KDN)      | `planner.experiment_design`             | `ExperimentDesigner._posterior_entropy_kdn()` |
| Marginal extraction          | `planner.experiment_design`             | `ExperimentDesigner.extract_marginal_samples()` |
| Instrumental noise           | `planner.instrument`                    | `InstrumentSimulator.add_noise()` |
| MCMC sampling (DREAM)        | `planner.mcmc_sampler`                  | `perform_mcmc()` |
| Single-realization evaluator | `planner.optimizer`                     | `_evaluate_single_realization()` |
| Sequential driver            | `planner.optimizer`                     | `optimize()` |
| Parallel driver              | `planner.optimizer`                     | `optimize_parallel()` |
| SLD contours                 | `planner.optimizer`                     | `_get_sld_contour()` |
| BIC / evidence               | `planner.experiment_design`             | `compute_bic()`, `compute_log_evidence()` |
| Model discrimination         | `planner.model_discriminator`           | `ModelDiscriminator.evaluate()` |
| Model loading (YAML/JSON)    | `planner.model_loader`                  | `load_experiment()`, `build_experiment()` |

---

## 9. References

- Treece, B. W., Kienzle, P. A., Hoogerheide, D. P., Majkrzak, C. F.,
  Lösche, M., & Heinrich, F. (2019). Optimization of reflectometry
  experiments using information theory. _J. Appl. Cryst._, **52**, 47–59.
  [doi:10.1107/S1600576718017016](https://doi.org/10.1107/S1600576718017016)

- ter Braak, C. J. F. (2006). A Markov chain Monte Carlo version of the
  genetic algorithm Differential Evolution: easy Bayesian computing for real
  parameter spaces. _Stat. Comput._, **16**, 239–249. (DREAM algorithm)

- Newton, M. A. & Raftery, A. E. (1994). Approximate Bayesian inference with
  the weighted likelihood bootstrap. _J. R. Stat. Soc. B_, **56**, 3–26.
  (Harmonic mean evidence estimator)

- Schwarz, G. (1978). Estimating the dimension of a model. _Ann. Statist._,
  **6**, 461–464. (Bayesian Information Criterion)
