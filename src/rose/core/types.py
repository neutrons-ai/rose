"""
Shared data types for ROSE.

Provides dataclasses used across the planner, modeler,
and web modules. These classes define the structure of
optimization results, model metadata, and instrument settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParameterInfo:
    """Metadata for a single model parameter.

    Attributes:
        name: Human-readable parameter name (e.g. "layer_a thickness").
        value: Current value of the parameter.
        min_bound: Lower bound of the parameter range.
        max_bound: Upper bound of the parameter range.
        fixed: Whether the parameter is held fixed during fitting.
    """

    name: str
    value: float
    min_bound: float
    max_bound: float
    fixed: bool = False


@dataclass
class RealizationData:
    """Results from a single noise realization.

    Attributes:
        q_values: Momentum transfer values.
        reflectivity: Clean (noiseless) reflectivity.
        noisy_reflectivity: Reflectivity with simulated noise.
        errors: Error bars on the noisy reflectivity.
        z: Depth values for the SLD profile.
        sld_best: Best-fit SLD profile.
        sld_low: Lower 90% confidence bound on SLD.
        sld_high: Upper 90% confidence bound on SLD.
        discrimination: Per-alternate discrimination metrics.
            Maps alternate model name to delta_metric value.
    """

    q_values: list[float] = field(default_factory=list)
    reflectivity: list[float] = field(default_factory=list)
    noisy_reflectivity: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    sld_best: list[float] = field(default_factory=list)
    sld_low: list[float] = field(default_factory=list)
    sld_high: list[float] = field(default_factory=list)
    discrimination: dict[str, float] = field(default_factory=dict)


@dataclass
class ParameterResult:
    """Optimization result for a single parameter value.

    Attributes:
        param_value: The tested parameter value.
        info_gain: Average information gain (bits) over realizations.
        info_gain_std: Standard deviation of information gain.
        realizations: Per-realization data.
        mean_discrimination: Mean ΔBIC (or log BF) per alternate model.
        model_probability: Mean P(primary | data) per alternate model.
    """

    param_value: float
    info_gain: float
    info_gain_std: float
    realizations: list[RealizationData] = field(default_factory=list)
    mean_discrimination: dict[str, float] = field(default_factory=dict)
    model_probability: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Full result of an optimization run.

    Attributes:
        parameter: Name of the parameter that was varied.
        parameter_values: All tested values.
        results: Per-value optimization results.
        optimal_value: Parameter value with highest information gain.
        max_information_gain: Highest information gain found.
        prior_entropy: Entropy of the prior distribution (bits).
        settings: Dictionary of run settings (steps, method, etc.).
        alternate_models: Names of alternate models tested.
        discrimination_method: Discrimination method used.
        discrimination_mode: Scoring mode used.
    """

    parameter: str
    parameter_values: list[float]
    results: list[ParameterResult]
    optimal_value: float
    max_information_gain: float
    prior_entropy: float
    settings: dict[str, object] = field(default_factory=dict)
    alternate_models: list[str] = field(default_factory=list)
    discrimination_method: str = ""
    discrimination_mode: str = ""
