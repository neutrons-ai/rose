"""
Model discrimination for Bayesian experiment design.

Compares a primary model against one or more alternate hypotheses
to assess whether a given experiment can distinguish them.  Two
discrimination metrics are supported:

- **BIC**: Bayesian Information Criterion difference
  (``ΔBIC = BIC_alt − BIC_primary``; positive favours primary).
- **evidence**: Log Bayes factor estimated via Newton–Raftery
  harmonic-mean estimator on existing DREAM chains.

Combined scoring allows experiments to be penalised when they
cannot distinguish the primary from simpler alternatives.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from refl1d.names import Experiment, FitProblem, Parameter

from . import mcmc_sampler
from .experiment_design import compute_bic, compute_log_evidence

logger = logging.getLogger(__name__)


def _set_param_on_sample(experiment: Experiment, name: str, value: float) -> bool:
    """Set a parameter by name on an experiment's sample, if it exists.

    Walks the layer stack parameters looking for a ``Parameter`` whose
    ``.name`` matches *name* and sets its value.  Returns ``True`` if
    the parameter was found and set, ``False`` otherwise (e.g. when the
    alternate model has removed the layer containing that parameter).
    """
    problem = FitProblem(experiment)
    for model in problem._models:
        for layer in model.parameters()["sample"]["layers"]:
            for _key, param in layer.items():
                if isinstance(param, dict):
                    for _sub, sub_param in param.items():
                        if isinstance(sub_param, Parameter) and sub_param.name == name:
                            sub_param.value = value
                            return True
                elif isinstance(param, Parameter) and param.name == name:
                    param.value = value
                    return True
    return False


class ModelDiscriminator:
    """Compare a primary model against alternate hypotheses.

    For each alternate model, fits it to the same noisy data used
    for the primary model and computes a discrimination metric.

    Args:
        alternate_experiments: List of ``(name, Experiment)`` tuples
            defining the alternate hypotheses.
        method: ``"bic"`` for BIC difference or ``"evidence"`` for
            harmonic-mean log Bayes factor.
    """

    def __init__(
        self,
        alternate_experiments: list[tuple[str, Experiment]],
        method: str = "bic",
    ):
        if method not in ("bic", "evidence"):
            raise ValueError(
                f"Unknown discrimination method '{method}'. Use 'bic' or 'evidence'."
            )
        self.alternate_experiments = alternate_experiments
        self.method = method

    def evaluate(
        self,
        primary_problem: FitProblem,
        primary_state: object,
        q_values: np.ndarray,
        noisy_reflectivity: np.ndarray,
        errors: np.ndarray,
        dq_values: np.ndarray,
        mcmc_steps: int = 1000,
        parallel: int = 1,
        save_problem: bool = False,
        param_to_optimize: str | None = None,
        param_value: float | None = None,
    ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        """Fit all alternate models and compute discrimination metrics.

        Args:
            primary_problem: The bumps ``FitProblem`` for the primary model.
            primary_state: DREAM state from the primary model's MCMC.
            q_values: Momentum transfer values.
            noisy_reflectivity: The noisy data (same for all models).
            errors: Error bars on the data.
            dq_values: Q resolution values.
            mcmc_steps: MCMC steps for alternate model fitting.
            parallel: bumps parallelism flag (1 = single CPU).
            save_problem: If ``True``, return serialised alternate
                ``FitProblem`` dicts alongside discrimination results.
            param_to_optimize: Name of the optimization parameter
                (e.g. ``"THF rho"``).  When provided together with
                *param_value*, the parameter is set on each alternate
                experiment before fitting so that the alternate model
                uses the same experimental condition as the primary.
            param_value: Value to assign to *param_to_optimize*.

        Returns:
            Tuple of ``(disc_results, alt_problems)`` where
            *disc_results* maps alternate name to
            ``{"delta_metric": float, "model_prob": float}`` and
            *alt_problems* maps alternate name to a serialised
            ``FitProblem`` dict (empty when *save_problem* is
            ``False``).
        """
        results: dict[str, dict[str, float]] = {}
        alt_problems: dict[str, Any] = {}

        for name, alt_experiment in self.alternate_experiments:
            try:
                # Apply the optimization parameter to the alternate
                # model so it uses the same experimental condition.
                if param_to_optimize is not None and param_value is not None:
                    found = _set_param_on_sample(
                        alt_experiment, param_to_optimize, param_value
                    )
                    if not found:
                        logger.debug(
                            "Optimization param '%s' not found on "
                            "alternate '%s' — skipped",
                            param_to_optimize,
                            name,
                        )

                alt_result, alt_mcmc_problem = mcmc_sampler.perform_mcmc(
                    alt_experiment.sample,
                    q_values,
                    noisy_reflectivity,
                    errors,
                    dq_values=dq_values,
                    mcmc_steps=mcmc_steps,
                    parallel=parallel,
                )
                alt_problem = FitProblem(alt_experiment)

                delta, prob = self.compute_discrimination(
                    primary_problem,
                    primary_state,
                    alt_problem,
                    alt_result.state,
                )
                results[name] = {
                    "delta_metric": delta,
                    "model_prob": prob,
                }

                if save_problem:
                    from bumps.serialize import serialize

                    alt_problems[name] = serialize(alt_mcmc_problem)

            except Exception as e:
                logger.error("Discrimination failed for alternate '%s': %s", name, e)
                logger.debug("Full traceback:", exc_info=True)
                results[name] = {
                    "delta_metric": float("nan"),
                    "model_prob": float("nan"),
                }

        return results, alt_problems

    def compute_discrimination(
        self,
        primary_problem: FitProblem,
        primary_state: object,
        alt_problem: FitProblem,
        alt_state: object,
    ) -> tuple[float, float]:
        """Compute the discrimination metric between two models.

        Args:
            primary_problem: FitProblem for the primary model.
            primary_state: DREAM state for the primary model.
            alt_problem: FitProblem for the alternate model.
            alt_state: DREAM state for the alternate model.

        Returns:
            ``(delta_metric, model_probability)`` where positive
            delta favours the primary model.
        """
        if self.method == "bic":
            bic_primary = compute_bic(primary_problem, primary_state)
            bic_alt = compute_bic(alt_problem, alt_state)
            delta = bic_alt - bic_primary  # positive = primary preferred
            prob = model_probability(delta, method="bic")
            return delta, prob

        # evidence method
        log_z_primary = compute_log_evidence(primary_state)
        log_z_alt = compute_log_evidence(alt_state)
        log_bf = log_z_primary - log_z_alt  # positive = primary preferred
        prob = model_probability(log_bf, method="evidence")
        return log_bf, prob


def model_probability(
    delta_metric: float,
    method: str = "bic",
) -> float:
    """Approximate probability of the primary model given data.

    Args:
        delta_metric: ΔBIC (for ``"bic"``) or log Bayes factor
            (for ``"evidence"``).  Positive values favour the
            primary model.
        method: ``"bic"`` or ``"evidence"``.

    Returns:
        Estimated P(primary | data) in [0, 1].
    """
    if np.isnan(delta_metric):
        return float("nan")

    if method == "bic":
        # BIC approximation: P ≈ 1 / (1 + exp(-ΔBIC / 2))
        return float(1.0 / (1.0 + np.exp(-delta_metric / 2.0)))

    # Evidence (log Bayes factor): BF = exp(log_bf)
    # P = BF / (1 + BF) = 1 / (1 + exp(-log_bf))
    return float(1.0 / (1.0 + np.exp(-delta_metric)))


def combine_scores(
    info_gain: float,
    model_probabilities: list[float],
    mode: str = "report",
) -> dict[str, Any]:
    """Combine information gain with model discrimination.

    Args:
        info_gain: Shannon information gain (bits).
        model_probabilities: P(primary | data) for each alternate.
        mode: ``"report"`` returns both metrics unchanged.
            ``"penalize"`` computes
            ``effective_info_gain = info_gain × mean(probs)``.

    Returns:
        Dict with ``"info_gain"``, ``"mean_model_prob"``, and
        (if penalize) ``"effective_info_gain"`` keys.
    """
    valid_probs = [p for p in model_probabilities if not np.isnan(p)]
    mean_prob = float(np.mean(valid_probs)) if valid_probs else float("nan")

    result: dict[str, Any] = {
        "info_gain": info_gain,
        "mean_model_prob": mean_prob,
    }

    if mode == "penalize" and not np.isnan(mean_prob):
        result["effective_info_gain"] = info_gain * mean_prob
    elif mode == "penalize":
        result["effective_info_gain"] = float("nan")

    return result
