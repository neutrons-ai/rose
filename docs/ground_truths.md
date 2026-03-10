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

- **Model files**: Python modules with `create_fit_experiment(q, dq, data, errors)` function returning a refl1d `Experiment` object
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
