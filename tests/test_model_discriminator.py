"""
Tests for rose.planner.model_discriminator.

Tests the model probability calculations, combine_scores logic,
and ModelDiscriminator validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rose.planner.model_discriminator import (
    ModelDiscriminator,
    _set_param_on_sample,
    combine_scores,
    model_probability,
)

# ── model_probability ────────────────────────────────────────────


class TestModelProbability:
    """Test the P(primary | data) sigmoid functions."""

    def test_bic_positive_delta_favours_primary(self):
        # Large positive ΔBIC → P close to 1
        p = model_probability(20.0, method="bic")
        assert p > 0.99

    def test_bic_negative_delta_favours_alternate(self):
        # Large negative ΔBIC → P close to 0
        p = model_probability(-20.0, method="bic")
        assert p < 0.01

    def test_bic_zero_delta_gives_half(self):
        p = model_probability(0.0, method="bic")
        assert abs(p - 0.5) < 1e-10

    def test_evidence_positive_favours_primary(self):
        p = model_probability(10.0, method="evidence")
        assert p > 0.99

    def test_evidence_negative_favours_alternate(self):
        p = model_probability(-10.0, method="evidence")
        assert p < 0.01

    def test_evidence_zero_gives_half(self):
        p = model_probability(0.0, method="evidence")
        assert abs(p - 0.5) < 1e-10

    def test_nan_input_returns_nan(self):
        p = model_probability(float("nan"), method="bic")
        assert math.isnan(p)

    def test_bic_monotonically_increasing(self):
        deltas = [-10, -5, 0, 5, 10]
        probs = [model_probability(d, method="bic") for d in deltas]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]


# ── combine_scores ───────────────────────────────────────────────


class TestCombineScores:
    """Test the combine_scores aggregation."""

    def test_report_mode_returns_both(self):
        result = combine_scores(2.5, [0.9, 0.8], mode="report")
        assert result["info_gain"] == 2.5
        assert abs(result["mean_model_prob"] - 0.85) < 1e-10
        assert "effective_info_gain" not in result

    def test_penalize_mode_computes_effective(self):
        result = combine_scores(2.0, [0.8, 0.6], mode="penalize")
        expected_eff = 2.0 * 0.7  # mean of [0.8, 0.6] = 0.7
        assert abs(result["effective_info_gain"] - expected_eff) < 1e-10

    def test_penalize_with_nan_prob_ignores_nans(self):
        result = combine_scores(3.0, [0.9, float("nan")], mode="penalize")
        assert abs(result["mean_model_prob"] - 0.9) < 1e-10
        assert abs(result["effective_info_gain"] - 2.7) < 1e-10

    def test_all_nan_probs_returns_nan(self):
        result = combine_scores(2.0, [float("nan")], mode="penalize")
        assert math.isnan(result["mean_model_prob"])
        assert math.isnan(result["effective_info_gain"])

    def test_empty_probs_returns_nan(self):
        result = combine_scores(2.0, [], mode="report")
        assert math.isnan(result["mean_model_prob"])

    def test_perfect_discrimination_preserves_gain(self):
        # P(primary) = 1.0 for all alternates → no penalty
        result = combine_scores(5.0, [1.0, 1.0], mode="penalize")
        assert abs(result["effective_info_gain"] - 5.0) < 1e-10

    def test_zero_discrimination_kills_gain(self):
        # P(primary) = 0 → effective gain = 0
        result = combine_scores(5.0, [0.0, 0.0], mode="penalize")
        assert abs(result["effective_info_gain"]) < 1e-10


# ── ModelDiscriminator construction ──────────────────────────────


class TestModelDiscriminatorConstruction:
    """Test ModelDiscriminator initialisation validation."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown discrimination method"):
            ModelDiscriminator([], method="magic")

    def test_valid_methods_accepted(self):
        for m in ("bic", "evidence"):
            md = ModelDiscriminator([], method=m)
            assert md.method == m

    def test_empty_alternates_ok(self):
        md = ModelDiscriminator([], method="bic")
        assert md.alternate_experiments == []


# ── _set_param_on_sample ─────────────────────────────────────────


class TestSetParamOnSample:
    """Test that optimization parameters get applied to alternate experiments."""

    def _make_experiment(self):
        from refl1d.names import SLD, Experiment, QProbe

        air = SLD("air", rho=0)
        film = SLD("film", rho=4.0)
        si = SLD("Si", rho=2.07)
        sample = air(0, 0) | film(50, 5) | si(0, 3)
        q = np.linspace(0.01, 0.2, 10)
        probe = QProbe(q, 0.02 * q)
        return Experiment(sample=sample, probe=probe)

    def test_sets_existing_material_param(self):
        expt = self._make_experiment()
        assert _set_param_on_sample(expt, "air rho", 99.0)
        # Verify the value was actually set
        from refl1d.names import FitProblem

        p = FitProblem(expt)
        for model in p._models:
            for layer in model.parameters()["sample"]["layers"]:
                for _key, param in layer.items():
                    if isinstance(param, dict):
                        for _sub, sp in param.items():
                            if hasattr(sp, "name") and sp.name == "air rho":
                                assert sp.value == 99.0

    def test_sets_thickness_param(self):
        expt = self._make_experiment()
        assert _set_param_on_sample(expt, "film thickness", 123.0)

    def test_returns_false_for_missing_param(self):
        expt = self._make_experiment()
        assert not _set_param_on_sample(expt, "nonexistent param", 1.0)
