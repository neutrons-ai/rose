"""Tests for rose.core.types."""

from rose.core.types import (
    OptimizationResult,
    ParameterInfo,
    ParameterResult,
    RealizationData,
)


def test_parameter_info_defaults():
    """Test ParameterInfo creation with defaults."""
    p = ParameterInfo(name="thickness", value=50.0, min_bound=10.0, max_bound=100.0)
    assert p.name == "thickness"
    assert p.fixed is False


def test_realization_data_defaults():
    """Test RealizationData initializes with empty lists."""
    r = RealizationData()
    assert r.q_values == []
    assert r.reflectivity == []


def test_optimization_result_round_trip():
    """Test building a full OptimizationResult."""
    result = OptimizationResult(
        parameter="layer_b thickness",
        parameter_values=[10.0, 20.0, 30.0],
        results=[
            ParameterResult(param_value=10.0, info_gain=0.3, info_gain_std=0.01),
            ParameterResult(param_value=20.0, info_gain=0.5, info_gain_std=0.02),
            ParameterResult(param_value=30.0, info_gain=0.4, info_gain_std=0.015),
        ],
        optimal_value=20.0,
        max_information_gain=0.5,
        prior_entropy=3.17,
    )
    assert result.optimal_value == 20.0
    assert len(result.results) == 3
    assert result.results[1].info_gain == 0.5
