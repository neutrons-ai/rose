"""
Tests for rose.planner.experiment_design.

Entropy calculations are tested with known data; refl1d-dependent
helpers are tested via the example model.
"""

from __future__ import annotations

import numpy as np
import pytest

from rose.planner.experiment_design import ExperimentDesigner

# ── fixtures using the real example model ────────────────────────


@pytest.fixture()
def designer():
    """ExperimentDesigner built from the layer_a_on_b example model."""
    from rose.planner.instrument import InstrumentSimulator
    from rose.planner.model_loader import load_experiment

    model_file = "examples/models/layer_a_on_b.yaml"
    sim = InstrumentSimulator()
    expt = load_experiment(model_file, sim.q_values, sim.dq_values)
    return ExperimentDesigner(expt, simulator=sim)


@pytest.fixture()
def designer_poi():
    """Designer with a subset of parameters_of_interest."""
    from rose.planner.instrument import InstrumentSimulator
    from rose.planner.model_loader import load_experiment

    model_file = "examples/models/layer_a_on_b.yaml"
    sim = InstrumentSimulator()
    expt = load_experiment(model_file, sim.q_values, sim.dq_values)
    return ExperimentDesigner(
        expt, simulator=sim, parameters_of_interest=["layer_A thickness"]
    )


# ── Prior entropy ────────────────────────────────────────────────


class TestPriorEntropy:
    def test_positive_entropy(self, designer):
        h = designer.prior_entropy()
        assert h > 0

    def test_entropy_changes_with_poi(self, designer, designer_poi):
        h_all = designer.prior_entropy()
        h_one = designer_poi.prior_entropy()
        # One parameter must have less entropy than all three
        assert h_one < h_all

    def test_entropy_manual_check(self, designer_poi):
        """layer_A thickness range is 10–100 → log₂(90)."""
        h = designer_poi.prior_entropy()
        expected = np.log2(90)
        assert abs(h - expected) < 0.01


# ── Posterior entropy (pure math, no refl1d) ─────────────────────


class TestPosteriorEntropy:
    """Test entropy calculations with synthetic MCMC-like arrays."""

    @staticmethod
    def _make_samples(n: int = 500, d: int = 3, scale: float = 1.0):
        rng = np.random.default_rng(42)
        return rng.normal(size=(n, d)) * scale

    def test_mvn_returns_float(self):
        samples = self._make_samples()
        h = ExperimentDesigner._posterior_entropy_mvn(samples)
        assert isinstance(h, float)

    def test_kdn_returns_float(self):
        samples = self._make_samples()
        h = ExperimentDesigner._posterior_entropy_kdn(samples)
        assert isinstance(h, float)

    def test_mvn_increases_with_spread(self):
        narrow = self._make_samples(scale=0.1)
        wide = self._make_samples(scale=10.0)
        h_narrow = ExperimentDesigner._posterior_entropy_mvn(narrow)
        h_wide = ExperimentDesigner._posterior_entropy_mvn(wide)
        assert h_wide > h_narrow

    def test_kdn_increases_with_spread(self):
        narrow = self._make_samples(scale=0.1)
        wide = self._make_samples(scale=10.0)
        h_narrow = ExperimentDesigner._posterior_entropy_kdn(narrow)
        h_wide = ExperimentDesigner._posterior_entropy_kdn(wide)
        assert h_wide > h_narrow

    def test_mvn_invalid_input_raises(self):
        with pytest.raises(ValueError):
            ExperimentDesigner._posterior_entropy_mvn(np.array([1.0]))

    def test_kdn_invalid_input_raises(self):
        with pytest.raises(ValueError):
            ExperimentDesigner._posterior_entropy_kdn(np.array([1.0]))

    def test_invalid_method_raises(self, designer):
        samples = self._make_samples()
        with pytest.raises(ValueError, match="Invalid entropy method"):
            designer.calculate_posterior_entropy(samples, method="bad")


# ── Marginal extraction ──────────────────────────────────────────


class TestMarginalSamples:
    def test_all_params_returns_full(self, designer):
        samples = np.random.default_rng(0).normal(size=(100, 3))
        result = designer.extract_marginal_samples(samples)
        assert result.shape == samples.shape

    def test_poi_returns_subset(self, designer_poi):
        samples = np.random.default_rng(0).normal(size=(100, 3))
        result = designer_poi.extract_marginal_samples(samples)
        assert result.shape[1] == 1  # only layer_A thickness


# ── set_parameter_to_optimize ────────────────────────────────────


class TestSetParameter:
    def test_set_valid_param(self, designer):
        designer.set_parameter_to_optimize("layer_B thickness", 99.0)
        assert designer.all_model_parameters["layer_B thickness"].value == 99.0

    def test_set_invalid_param_raises(self, designer):
        with pytest.raises(ValueError, match="not found"):
            designer.set_parameter_to_optimize("nonexistent", 1.0)


# ── draw_truth_from_prior / restore_parameter_values ─────────────


class TestDrawTruthFromPrior:
    def test_drawn_values_within_bounds(self, designer):
        """All drawn values must lie within their fit bounds."""
        rng = np.random.default_rng(42)
        drawn = designer.draw_truth_from_prior(rng=rng)
        for name, val in drawn.items():
            pmin, pmax = designer.parameters[name]["bounds"]
            assert pmin <= val <= pmax, f"{name}={val} outside [{pmin}, {pmax}]"

    def test_drawn_values_differ_from_original(self, designer):
        """With high probability, at least one drawn value differs."""
        original = {p.name: p.value for p in designer.problem.parameters}
        rng = np.random.default_rng(123)
        drawn = designer.draw_truth_from_prior(rng=rng)
        # Very unlikely all three match exactly
        assert any(abs(drawn[n] - original[n]) > 1e-10 for n in drawn)

    def test_model_updated_after_draw(self, designer):
        """After drawing, the model's parameter values should match the draw."""
        rng = np.random.default_rng(7)
        drawn = designer.draw_truth_from_prior(rng=rng)
        for param in designer.problem.parameters:
            assert abs(param.value - drawn[param.name]) < 1e-12

    def test_restore_returns_to_original(self, designer):
        """restore_parameter_values brings params back to saved values."""
        saved = {p.name: p.value for p in designer.problem.parameters}
        designer.draw_truth_from_prior()
        designer.restore_parameter_values(saved)
        for param in designer.problem.parameters:
            assert abs(param.value - saved[param.name]) < 1e-12

    def test_successive_draws_differ(self, designer):
        """Two draws with different seeds produce different truths."""
        rng1 = np.random.default_rng(1)
        draw1 = designer.draw_truth_from_prior(rng=rng1)
        rng2 = np.random.default_rng(999)
        draw2 = designer.draw_truth_from_prior(rng=rng2)
        assert any(abs(draw1[n] - draw2[n]) > 1e-6 for n in draw1)

    def test_default_rng_when_none(self, designer):
        """Passing no rng still works (creates a fresh generator)."""
        drawn = designer.draw_truth_from_prior()
        assert len(drawn) == len(list(designer.problem.parameters))


# ── repr ─────────────────────────────────────────────────────────


def test_repr(designer):
    r = repr(designer)
    assert "ExperimentDesigner" in r
    assert "layer_A" in r
