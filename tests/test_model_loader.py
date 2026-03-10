"""
Tests for rose.planner.model_loader.

Uses the real example YAML models in examples/models/.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from rose.planner.model_loader import (
    build_experiment,
    inspect_model,
    load_experiment,
    load_model_description,
)


@pytest.fixture()
def q_dq():
    q = np.logspace(np.log10(0.008), np.log10(0.2), 30)
    return q, 0.025 * q


# ── load_model_description ───────────────────────────────────────


class TestLoadModelDescription:
    def test_load_yaml(self):
        desc = load_model_description("examples/models/layer_a_on_b.yaml")
        assert "layers" in desc
        assert len(desc["layers"]) == 4

    def test_load_json(self, tmp_path):
        data = {
            "layers": [
                {"name": "air", "rho": 0},
                {"name": "Si", "rho": 2.07},
            ]
        }
        jf = tmp_path / "model.json"
        jf.write_text(json.dumps(data))
        desc = load_model_description(str(jf))
        assert len(desc["layers"]) == 2

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "model.txt"
        bad.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            load_model_description(str(bad))

    def test_missing_layers_raises(self, tmp_path):
        bad = tmp_path / "model.yaml"
        bad.write_text("name: no layers here\n")
        with pytest.raises(ValueError, match="layers"):
            load_model_description(str(bad))

    def test_too_few_layers_raises(self, tmp_path):
        bad = tmp_path / "model.yaml"
        bad.write_text(yaml.dump({"layers": [{"name": "only_one"}]}))
        with pytest.raises(ValueError, match="at least 2"):
            load_model_description(str(bad))

    def test_invalid_fit_bounds_raises(self, tmp_path):
        bad = tmp_path / "model.yaml"
        bad.write_text(
            yaml.dump(
                {
                    "layers": [
                        {"name": "air", "rho": 0},
                        {"name": "Si", "rho": 2.07, "fit": {"rho": [5, 1]}},
                    ]
                }
            )
        )
        with pytest.raises(ValueError, match="min"):
            load_model_description(str(bad))


# ── load_experiment ──────────────────────────────────────────────


class TestLoadExperiment:
    def test_load_layer_a_on_b(self, q_dq):
        q, dq = q_dq
        expt = load_experiment("examples/models/layer_a_on_b.yaml", q, dq)
        assert hasattr(expt, "reflectivity")
        assert hasattr(expt, "sample")

    def test_reflectivity_computable(self, q_dq):
        q, dq = q_dq
        expt = load_experiment("examples/models/layer_a_on_b.yaml", q, dq)
        qvals, r = expt.reflectivity()
        assert len(r) == len(q)
        assert np.all(np.isfinite(r))

    def test_file_not_found_raises(self, q_dq):
        q, dq = q_dq
        with pytest.raises(FileNotFoundError):
            load_experiment("nonexistent_model.yaml", q, dq)

    def test_cu_thf_loads(self, q_dq):
        q, dq = q_dq
        expt = load_experiment("examples/models/cu_thf.yaml", q, dq)
        _, r = expt.reflectivity()
        assert len(r) == len(q)


# ── build_experiment ─────────────────────────────────────────────


class TestBuildExperiment:
    def test_build_from_dict(self, q_dq):
        q, dq = q_dq
        desc = {
            "layers": [
                {"name": "air", "rho": 0},
                {
                    "name": "stuff",
                    "rho": 4.0,
                    "thickness": 50,
                    "interface": 5,
                    "fit": {"thickness": [10, 100]},
                },
                {"name": "Si", "rho": 2.07},
            ]
        }
        expt = build_experiment(desc, q, dq)
        _, r = expt.reflectivity()
        assert len(r) == len(q)

    def test_unknown_fit_key_raises(self, q_dq):
        q, dq = q_dq
        desc = {
            "layers": [
                {"name": "air", "rho": 0},
                {"name": "x", "rho": 3, "fit": {"banana": [1, 10]}},
                {"name": "Si", "rho": 2.07},
            ]
        }
        with pytest.raises(ValueError, match="Unknown fit parameter"):
            build_experiment(desc, q, dq)


# ── inspect_model ────────────────────────────────────────────────


class TestInspectModel:
    def test_inspect_returns_variable_params(self):
        info = inspect_model("examples/models/layer_a_on_b.yaml")
        assert len(info["variable"]) >= 3
        names = [p["name"] for p in info["variable"]]
        assert "layer_A thickness" in names

    def test_inspect_has_fixed_params(self):
        info = inspect_model("examples/models/layer_a_on_b.yaml")
        assert isinstance(info["fixed"], list)

    def test_inspect_variable_have_bounds(self):
        info = inspect_model("examples/models/layer_a_on_b.yaml")
        for p in info["variable"]:
            lo, hi = p["bounds"]
            assert hi > lo
