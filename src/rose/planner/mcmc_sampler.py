"""
MCMC posterior sampling using the bumps DREAM algorithm.

Wraps the bumps ``fit`` function to run DREAM on a refl1d
sample with synthetic noisy data.
"""

from __future__ import annotations

import numpy as np
from bumps.fitters import fit
from refl1d.names import Experiment, FitProblem, QProbe


def perform_mcmc(
    sample: object,
    q_values: np.ndarray,
    noisy_reflectivity: np.ndarray,
    errors: np.ndarray,
    dq_values: np.ndarray,
    mcmc_steps: int = 1000,
    burn_steps: int = 1000,
    parallel: int = 0,
):
    """Run MCMC analysis on synthetic reflectivity data.

    Creates a fresh ``FitProblem`` from the given sample and noisy data,
    then runs the DREAM sampler.

    Args:
        sample: refl1d sample object (layer stack).
        q_values: Momentum transfer values.
        noisy_reflectivity: Noisy reflectivity data to fit.
        errors: Error bars on the reflectivity.
        dq_values: Q resolution values.
        mcmc_steps: Number of MCMC steps after burn-in.
        burn_steps: Number of burn-in steps to discard.
        parallel: Number of CPUs for bumps DREAM. ``0`` uses all
            available CPUs, ``1`` disables parallelism.

    Returns:
        A ``(result, problem)`` tuple where *result* is a bumps fit
        result whose ``.state`` contains the DREAM chain and *problem*
        is the ``FitProblem`` used for fitting.
    """
    probe = QProbe(q_values, dq_values, R=noisy_reflectivity, dR=errors)
    expt = Experiment(sample=sample, probe=probe)
    problem = FitProblem(expt)
    problem.model_update()

    result = fit(
        problem,
        method="dream",
        samples=mcmc_steps,
        burn=burn_steps,
        verbose=0,
        parallel=parallel,
    )

    result.state.keep_best()
    result.state.mark_outliers()

    return result, problem
