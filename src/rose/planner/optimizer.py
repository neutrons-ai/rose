"""
Optimization driver for experiment design.

Evaluates information gain across a grid of parameter values,
either sequentially or in parallel using ``ProcessPoolExecutor``.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from refl1d import uncertainty
from tqdm import tqdm

from . import mcmc_sampler
from .experiment_design import ExperimentDesigner

logger = logging.getLogger(__name__)

#: Hard cap on the number of parameter values to evaluate in a single run.
MAX_PARAM_VALUES = 200

#: Maximum number of parallel worker processes.
MAX_WORKERS = 8


# ------------------------------------------------------------------
# SLD contour helper (ported from analyzer_tools.utils.model_utils)
# ------------------------------------------------------------------


def _get_sld_contour(
    problem,
    state,
    cl: int = 90,
    npoints: int = 200,
    trim: int = 1000,
    portion: float = 0.3,
    index: int = 1,
    align: str | int = "auto",
) -> list[np.ndarray]:
    """Extract SLD depth-profile contours from MCMC state.

    Args:
        problem: A bumps ``FitProblem``.
        state: The DREAM sampler state.
        cl: Confidence level for the bands.
        npoints: Number of depth points.
        trim: Keep only the last *trim* samples.
        portion: Fraction of the chain to use.
        index: Profile group index (1 = rho).
        align: Alignment mode for stacking profiles.

    Returns:
        List of arrays ``[z, best, low, high]`` per model.
    """
    points, _logp = state.sample(portion=portion)
    points = points[-trim:]
    original = problem.getp()
    _profiles, slabs, _Q, _residuals = uncertainty.calc_errors(problem, points)
    problem.setp(original)

    profiles = uncertainty.align_profiles(_profiles, slabs, align)

    contours = []
    for _model, group in profiles.items():
        z = np.hstack([line[0] for line in group])
        zp = np.linspace(np.min(z), np.max(z), npoints)
        data, _cols = uncertainty._build_profile_matrix(group, index, zp, [cl])
        contours.append(data)
    return contours


# ------------------------------------------------------------------
# Single parameter-value evaluator
# ------------------------------------------------------------------


def evaluate_param(
    designer: ExperimentDesigner,
    param_to_optimize: str,
    value: float,
    realizations: int,
    prior_entropy: float,
    mcmc_steps: int,
    entropy_method: str,
) -> tuple[float, float, float, list[dict[str, Any]]]:
    """Evaluate information gain for a single parameter value.

    For each noise realization:
      1. Draw 'true' fitted-parameter values from the prior.
      2. Simulate noisy reflectivity from that truth.
      3. Restore original parameter values (MCMC starting point).
      4. Run MCMC to sample the posterior.
      5. Compute posterior entropy.
      6. Accumulate ``ΔH = H_prior − H_posterior``.

    Drawing from the prior ensures the information gain is an
    expectation over unknown true values, following Treece et al.

    A per-call ``np.random.Generator`` seeded from *value* ensures
    reproducibility and avoids correlated draws across parallel workers.

    Args:
        designer: Configured ``ExperimentDesigner``.
        param_to_optimize: Name of the variable to set.
        value: Value to assign before each realization.
        realizations: Number of noise realizations.
        prior_entropy: Pre-computed prior entropy (bits).
        mcmc_steps: MCMC chain length.
        entropy_method: ``"mvn"`` or ``"kdn"``.

    Returns:
        ``(value, mean_info_gain, std_info_gain, realization_data)``
    """
    # Seed from the float value so each worker gets a unique but
    # reproducible stream — avoids correlated draws across processes.
    rng = np.random.default_rng(seed=abs(hash(value)))

    designer.set_parameter_to_optimize(param_to_optimize, value)

    # Save the original (YAML) values for fitted parameters so we can
    # restore them as the MCMC starting point after each truth draw.
    original_values = {p.name: p.value for p in designer.problem.parameters}

    realization_gains: list[float] = []
    realization_data: list[dict[str, Any]] = []

    for _ in range(realizations):
        try:
            # Draw random "true" values from the prior for the synthetic data
            designer.draw_truth_from_prior(rng=rng)
            q_values, r_calc = designer.experiment.reflectivity()
            noisy_reflectivity, errors = designer.simulator.add_noise(r_calc, rng=rng)

            # Restore original values so MCMC starts from the YAML values
            designer.restore_parameter_values(original_values)

            mcmc_result = mcmc_sampler.perform_mcmc(
                designer.experiment.sample,
                q_values,
                noisy_reflectivity,
                errors,
                dq_values=designer.simulator.dq_values,
                mcmc_steps=mcmc_steps,
            )
            mcmc_samples = mcmc_result.state.draw().points

            marginal = designer.extract_marginal_samples(mcmc_samples)
            posterior_entropy = designer.calculate_posterior_entropy(
                marginal, method=entropy_method
            )
            info_gain = prior_entropy - posterior_entropy
            realization_gains.append(info_gain)

            # SLD contour
            z, best, low, high = _get_sld_contour(
                designer.problem,
                mcmc_result.state,
                cl=90,
                npoints=200,
                index=1,
                align=-1,
            )[0]

            # Best-fit reflectivity
            best_p, _ = mcmc_result.state.best()
            designer.problem.setp(best_p)
            _, fit_reflectivity = designer.experiment.reflectivity()

            realization_data.append(
                {
                    "q_values": q_values.tolist(),
                    "dq_values": designer.simulator.dq_values.tolist(),
                    "reflectivity": fit_reflectivity.tolist(),
                    "noisy_reflectivity": noisy_reflectivity.tolist(),
                    "errors": errors.tolist(),
                    "z": z.tolist() if hasattr(z, "tolist") else list(z),
                    "sld_best": best.tolist()
                    if hasattr(best, "tolist")
                    else list(best),
                    "sld_low": low.tolist() if hasattr(low, "tolist") else list(low),
                    "sld_high": high.tolist()
                    if hasattr(high, "tolist")
                    else list(high),
                    "posterior_entropy": posterior_entropy,
                }
            )

        except Exception as e:
            logger.error("Realization failed: %s", e)
            logger.debug("Full traceback:", exc_info=True)
            realization_gains.append(float("nan"))
        finally:
            designer.restore_parameter_values(original_values)

    n_failed = sum(1 for g in realization_gains if np.isnan(g))
    if n_failed > 0:
        logger.warning(
            "%d of %d realizations failed for value %.3f",
            n_failed,
            realizations,
            value,
        )

    avg = float(np.nanmean(realization_gains))
    std = float(np.nanstd(realization_gains))
    return value, avg, std, realization_data


# ------------------------------------------------------------------
# Sequential & parallel drivers
# ------------------------------------------------------------------


def optimize(
    designer: ExperimentDesigner,
    param_to_optimize: str,
    param_values: list[float],
    realizations: int = 3,
    mcmc_steps: int = 2000,
    entropy_method: str = "kdn",
) -> tuple[list[list[float]], list[list[dict[str, Any]]]]:
    """Run optimization sequentially.

    Args:
        designer: Configured experiment designer.
        param_to_optimize: Parameter name to vary.
        param_values: Grid of values to test.
        realizations: Noise realizations per value.
        mcmc_steps: MCMC chain length.
        entropy_method: ``"mvn"`` or ``"kdn"``.

    Returns:
        ``(results, simulated_data)`` where *results* is a list of
        ``[value, info_gain, std]`` and *simulated_data* is a list of
        per-value realization dicts.
    """
    if param_to_optimize not in designer.all_model_parameters:
        raise ValueError(
            f"Parameter '{param_to_optimize}' not found in model parameters"
        )
    if len(param_values) > MAX_PARAM_VALUES:
        raise ValueError(
            f"Too many parameter values ({len(param_values)}); "
            f"max is {MAX_PARAM_VALUES}"
        )

    prior_entropy = designer.prior_entropy()
    logger.info("Prior entropy: %.4f bits", prior_entropy)

    results: list[list[float]] = []
    simulated_data: list[list[dict[str, Any]]] = []

    for value in tqdm(param_values, desc="Optimizing", unit="val"):
        try:
            val, gain, std, rdata = evaluate_param(
                designer,
                param_to_optimize,
                value,
                realizations,
                prior_entropy,
                mcmc_steps,
                entropy_method,
            )
            results.append([val, gain, std])
            simulated_data.append(rdata)
            logger.info("Value %.3f: ΔH = %.4f ± %.4f bits", val, gain, std)
        except Exception as e:
            logger.error("Error evaluating value %s: %s", value, e)
            logger.debug("Full traceback:", exc_info=True)

    return results, simulated_data


def optimize_parallel(
    designer: ExperimentDesigner,
    param_to_optimize: str,
    param_values: list[float],
    realizations: int = 3,
    mcmc_steps: int = 2000,
    entropy_method: str = "kdn",
) -> tuple[list[list[float]], list[list[dict[str, Any]]]]:
    """Run optimization in parallel using ``ProcessPoolExecutor``.

    Same interface as :func:`optimize` but fans out work across CPUs.
    Results are returned in the original *param_values* order.
    """
    if param_to_optimize not in designer.all_model_parameters:
        raise ValueError(
            f"Parameter '{param_to_optimize}' not found in model parameters"
        )
    if len(param_values) > MAX_PARAM_VALUES:
        raise ValueError(
            f"Too many parameter values ({len(param_values)}); "
            f"max is {MAX_PARAM_VALUES}"
        )

    prior_entropy = designer.prior_entropy()
    logger.info("Prior entropy: %.4f bits", prior_entropy)

    results: list[tuple[float, float, float]] = []
    simulated_data: list[list[dict[str, Any]]] = []

    max_workers = min(len(param_values), os.cpu_count() or 4, MAX_WORKERS)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_value = {
            executor.submit(
                evaluate_param,
                designer,
                param_to_optimize,
                value,
                realizations,
                prior_entropy,
                mcmc_steps,
                entropy_method,
            ): value
            for value in param_values
        }

        for future in tqdm(
            as_completed(future_to_value),
            total=len(param_values),
            desc="Optimizing",
            unit="val",
        ):
            value = future_to_value[future]
            try:
                val, gain, std, rdata = future.result()
                results.append((val, gain, std))
                simulated_data.append(rdata)
                logger.info("Value %.3f: ΔH = %.4f ± %.4f bits", val, gain, std)
            except Exception as e:
                logger.error("Error evaluating value %s: %s", value, e)
                logger.debug("Full traceback:", exc_info=True)

    # Re-order to match input param_values
    value_to_result = {r[0]: r for r in results}
    value_to_data = {r[0]: d for r, d in zip(results, simulated_data)}
    ordered_results = [
        [v, value_to_result[v][1], value_to_result[v][2]]
        for v in param_values
        if v in value_to_result
    ]
    ordered_data = [value_to_data[v] for v in param_values if v in value_to_data]

    return ordered_results, ordered_data
