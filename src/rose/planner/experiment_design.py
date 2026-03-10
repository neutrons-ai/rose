"""
Experimental design optimization for neutron reflectometry.

Implements the Bayesian framework for optimizing neutron reflectometry
experiments by maximizing expected Shannon information gain, following
Treece et al., J. Appl. Cryst. (2019), 52, 47-59.
"""

from __future__ import annotations

import logging

import numpy as np
from refl1d.names import Experiment, FitProblem, Parameter
from scipy.stats import gaussian_kde, multivariate_normal

from . import instrument as inst

logger = logging.getLogger(__name__)


class ExperimentDesigner:
    """Suggest optimal experimental protocols for neutron reflectometry.

    Maximizes the expected Shannon information gain by computing
    the difference between prior and posterior entropy over one or more
    noise realizations for each candidate parameter value.

    Args:
        experiment: A refl1d ``Experiment`` object defining the sample.
        simulator: An ``InstrumentSimulator`` for generating synthetic noise.
        parameters_of_interest: Optional subset of parameter names.
            If provided, only these parameters contribute to entropy
            calculations (marginalized distribution). When ``None``,
            all variable parameters are used.

    Example:
        >>> designer = ExperimentDesigner(experiment, simulator,
        ...     parameters_of_interest=["layer_a thickness"])
        >>> designer.prior_entropy()
        3.17
    """

    def __init__(
        self,
        experiment: Experiment,
        simulator: inst.InstrumentSimulator,
        parameters_of_interest: list[str] | None = None,
    ):
        self.experiment = experiment
        self.problem = FitProblem(experiment)
        self.parameters_of_interest: list[str] | None = parameters_of_interest
        if not parameters_of_interest:
            self.parameters_of_interest = None

        # Variable (fitted) parameters
        self.all_model_parameters = self._model_parameters_to_dict()
        self.parameters = self._get_parameters()

        self.simulator = simulator or inst.InstrumentSimulator()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [f"ExperimentDesigner with {len(self.problem.parameters)} parameters"]
        header = f"  {'name':<25} {'value':<10} {'bounds':<20} {'H_prior':<10}"
        lines.append(header)
        for name, p in self.parameters.items():
            star = "*" if p["is_of_interest"] else ""
            label = f"{name}{star}"
            lines.append(
                f"  {label:<25} {p['value']:<10.4g} "
                f"{str(p['bounds']):<20} {p['h_prior']:<10.4g}"
            )
        return "\n".join(lines)

    def set_parameter_to_optimize(self, param_name: str, value: float) -> None:
        """Set a fixed parameter value on the underlying model.

        This modifies a parameter that is *not* one of the free (fitted)
        parameters — typically the controllable experimental variable
        whose optimal value we are searching for.

        Args:
            param_name: Name of the model parameter.
            value: New value to assign.

        Raises:
            ValueError: If *param_name* is not found in the model.
        """
        if param_name not in self.all_model_parameters:
            raise ValueError(
                f"Parameter '{param_name}' not found in model. "
                f"Available: {list(self.all_model_parameters.keys())}"
            )
        self.all_model_parameters[param_name].value = value
        self.problem.model_update()

    def prior_entropy(self) -> float:
        """Shannon entropy of the prior distribution (bits).

        Computed as the sum of ``log2(max - min)`` for each parameter
        of interest (uniform prior assumption).

        Returns:
            Entropy in bits.

        Raises:
            ValueError: If any parameter has undefined or invalid bounds.
        """
        h_prior = 0.0
        for name, p in self.parameters.items():
            if not p["is_of_interest"]:
                continue
            pmin, pmax = p["bounds"]
            if pmin is None or pmax is None:
                raise ValueError(f"Parameter '{name}' has undefined bounds")
            if pmax <= pmin:
                raise ValueError(
                    f"Parameter '{name}' has invalid bounds: {pmin} >= {pmax}"
                )
            p["h_prior"] = float(np.log2(pmax - pmin))
            h_prior += p["h_prior"]
        return h_prior

    # ------------------------------------------------------------------
    # Posterior entropy
    # ------------------------------------------------------------------

    def extract_marginal_samples(self, mcmc_samples: np.ndarray) -> np.ndarray:
        """Extract columns for parameters of interest from MCMC samples.

        Args:
            mcmc_samples: Full MCMC sample array (n_samples x n_params).

        Returns:
            Sub-array with only the columns for parameters of interest.
        """
        if not self.parameters_of_interest:
            return mcmc_samples

        all_names = [p.name for p in self.problem.parameters]
        indices = []
        for name in self.parameters_of_interest:
            if name in all_names:
                indices.append(all_names.index(name))
            else:
                logger.warning("Parameter '%s' not found in MCMC samples", name)

        if not indices:
            logger.debug("No parameters of interest found; using all parameters")
            return mcmc_samples

        return mcmc_samples[:, indices]

    def calculate_posterior_entropy(
        self, mcmc_samples: np.ndarray, method: str = "kdn"
    ) -> float:
        """Compute posterior entropy from MCMC samples.

        Args:
            mcmc_samples: 2-D array (n_samples x n_params).
            method: ``"mvn"`` for multivariate-normal approximation,
                ``"kdn"`` for kernel density estimation.

        Returns:
            Posterior entropy in bits.

        Raises:
            ValueError: On invalid *method* or degenerate samples.
        """
        method = method.lower()
        if method == "mvn":
            return self._posterior_entropy_mvn(mcmc_samples)
        elif method == "kdn":
            return self._posterior_entropy_kdn(mcmc_samples)
        else:
            raise ValueError(f"Invalid entropy method '{method}'. Use 'mvn' or 'kdn'.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_parameters(self) -> dict[str, dict]:
        """Build a dict of variable parameters with metadata."""
        parameters: dict[str, dict] = {}
        for param in self.problem.parameters:
            is_of_interest = (
                self.parameters_of_interest is None
                or param.name in self.parameters_of_interest
            )
            parameters[param.name] = {
                "name": param.name,
                "value": param.value,
                "bounds": param.bounds,
                "is_of_interest": is_of_interest,
                "h_prior": 0.0,
            }

        if self.parameters_of_interest:
            missing = [
                n
                for n in self.parameters_of_interest
                if n not in self.all_model_parameters
            ]
            if missing:
                logger.warning("Parameters %s not found in model", missing)
                logger.info(
                    "Available parameters: %s",
                    list(self.all_model_parameters.keys()),
                )
        return parameters

    def _model_parameters_to_dict(self) -> dict[str, Parameter]:
        """Flatten model parameters into ``{name: Parameter}``."""
        param_dict: dict[str, Parameter] = {}
        models = self.problem._models
        if len(models) != 1:
            raise ValueError(
                f"Expected exactly one model in the problem, found {len(models)}"
            )
        struct_dict = models[0].parameters()["sample"]["layers"]
        for layer in struct_dict:
            for _key, param in layer.items():
                if isinstance(param, dict):
                    for _sub_key, sub_value in param.items():
                        param_dict[sub_value.name] = sub_value
                else:
                    param_dict[param.name] = param
        return param_dict

    @staticmethod
    def _posterior_entropy_mvn(mcmc_samples: np.ndarray) -> float:
        """Multivariate-normal posterior entropy (bits)."""
        if mcmc_samples.ndim != 2 or mcmc_samples.shape[0] < 2:
            raise ValueError("MCMC samples must be a 2-D array with at least 2 rows.")
        try:
            cov = np.cov(mcmc_samples, rowvar=False)
            entropy_nats = multivariate_normal.entropy(cov=cov)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix; adding regularisation")
            cov = np.cov(mcmc_samples, rowvar=False)
            cov += 1e-10 * np.eye(cov.shape[0])
            entropy_nats = multivariate_normal.entropy(cov=cov)
        return float(entropy_nats / np.log(2))

    @staticmethod
    def _posterior_entropy_kdn(mcmc_samples: np.ndarray) -> float:
        """Kernel density estimation posterior entropy (bits)."""
        if mcmc_samples.ndim != 2 or mcmc_samples.shape[0] < 2:
            raise ValueError("MCMC samples must be a 2-D array with at least 2 rows.")
        try:
            kde = gaussian_kde(mcmc_samples.T)
            log_probs = kde.logpdf(mcmc_samples.T)
            entropy_nats = -np.mean(log_probs)
        except Exception:
            logger.warning("KDE failed; falling back to MVN entropy")
            logger.debug("KDE failure traceback:", exc_info=True)
            cov = np.cov(mcmc_samples, rowvar=False)
            cov += 1e-10 * np.eye(cov.shape[0])
            entropy_nats = multivariate_normal.entropy(cov=cov)
        return float(entropy_nats / np.log(2))
