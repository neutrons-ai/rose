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
    build_alternate_descriptions,
    build_alternate_experiments,
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


# ── experiment & optimization validation ─────────────────────────


class TestExperimentValidation:
    def test_valid_experiment_section(self):
        desc = load_model_description("examples/models/layer_a_on_b.yaml")
        expt = desc.get("experiment", {})
        assert expt["q_min"] == 0.008
        assert expt["q_points"] == 50

    def test_unknown_experiment_key_raises(self, tmp_path):
        bad = tmp_path / "model.yaml"
        bad.write_text(
            yaml.dump(
                {
                    "layers": [{"name": "air", "rho": 0}, {"name": "Si", "rho": 2.07}],
                    "experiment": {"banana": 42},
                }
            )
        )
        with pytest.raises(ValueError, match="Unknown keys.*experiment"):
            load_model_description(str(bad))

    def test_q_min_gte_q_max_raises(self, tmp_path):
        bad = tmp_path / "model.yaml"
        bad.write_text(
            yaml.dump(
                {
                    "layers": [{"name": "air", "rho": 0}, {"name": "Si", "rho": 2.07}],
                    "experiment": {"q_min": 0.5, "q_max": 0.1},
                }
            )
        )
        with pytest.raises(ValueError, match="q_min"):
            load_model_description(str(bad))

    def test_q_points_out_of_range_raises(self, tmp_path):
        bad = tmp_path / "model.yaml"
        bad.write_text(
            yaml.dump(
                {
                    "layers": [{"name": "air", "rho": 0}, {"name": "Si", "rho": 2.07}],
                    "experiment": {"q_points": 2},
                }
            )
        )
        with pytest.raises(ValueError, match="q_points"):
            load_model_description(str(bad))


class TestOptimizationValidation:
    def test_valid_optimization_section(self):
        desc = load_model_description("examples/models/layer_a_on_b.yaml")
        opt = desc["optimization"]
        assert opt["param"] == "layer_B thickness"
        assert isinstance(opt["param_values"], list)
        assert len(opt["param_values"]) > 0

    def test_unknown_optimization_key_raises(self, tmp_path):
        bad = tmp_path / "model.yaml"
        bad.write_text(
            yaml.dump(
                {
                    "layers": [{"name": "air", "rho": 0}, {"name": "Si", "rho": 2.07}],
                    "optimization": {"nonsense": True},
                }
            )
        )
        with pytest.raises(ValueError, match="Unknown keys.*optimization"):
            load_model_description(str(bad))

    def test_bad_entropy_method_raises(self, tmp_path):
        bad = tmp_path / "model.yaml"
        bad.write_text(
            yaml.dump(
                {
                    "layers": [{"name": "air", "rho": 0}, {"name": "Si", "rho": 2.07}],
                    "optimization": {"entropy_method": "bogus"},
                }
            )
        )
        with pytest.raises(ValueError, match="entropy_method"):
            load_model_description(str(bad))

    def test_empty_param_values_raises(self, tmp_path):
        bad = tmp_path / "model.yaml"
        bad.write_text(
            yaml.dump(
                {
                    "layers": [{"name": "air", "rho": 0}, {"name": "Si", "rho": 2.07}],
                    "optimization": {"param_values": []},
                }
            )
        )
        with pytest.raises(ValueError, match="param_values"):
            load_model_description(str(bad))


# ── alternate models validation ──────────────────────────────────


class TestAlternateModelsValidation:
    """Test validation of alternate_models in the optimization section."""

    def _base_desc(self):
        return {
            "layers": [
                {"name": "air", "rho": 0},
                {"name": "oxide", "rho": 5.0, "thickness": 20, "interface": 3},
                {"name": "Cu", "rho": 6.3},
            ],
        }

    def test_valid_remove_action(self, tmp_path):
        desc = self._base_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "no_oxide",
                    "modifications": [{"action": "remove", "layer": "oxide"}],
                }
            ]
        }
        f = tmp_path / "model.yaml"
        f.write_text(yaml.dump(desc))
        loaded = load_model_description(str(f))
        alts = loaded["optimization"]["alternate_models"]
        assert len(alts) == 1

    def test_valid_modify_action(self, tmp_path):
        desc = self._base_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "thick_oxide",
                    "modifications": [
                        {
                            "action": "modify",
                            "layer": "oxide",
                            "set": {"thickness": 50},
                        }
                    ],
                }
            ]
        }
        f = tmp_path / "model.yaml"
        f.write_text(yaml.dump(desc))
        loaded = load_model_description(str(f))
        assert loaded["optimization"]["alternate_models"][0]["name"] == "thick_oxide"

    def test_invalid_action_raises(self, tmp_path):
        desc = self._base_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "bad",
                    "modifications": [{"action": "explode", "layer": "oxide"}],
                }
            ]
        }
        f = tmp_path / "model.yaml"
        f.write_text(yaml.dump(desc))
        with pytest.raises(ValueError, match="action"):
            load_model_description(str(f))

    def test_missing_name_raises(self, tmp_path):
        desc = self._base_desc()
        desc["optimization"] = {
            "alternate_models": [
                {"modifications": [{"action": "remove", "layer": "oxide"}]}
            ]
        }
        f = tmp_path / "model.yaml"
        f.write_text(yaml.dump(desc))
        with pytest.raises(ValueError, match="name"):
            load_model_description(str(f))

    def test_unknown_layer_raises(self, tmp_path):
        desc = self._base_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "bad_ref",
                    "modifications": [
                        {"action": "remove", "layer": "nonexistent_layer"}
                    ],
                }
            ]
        }
        f = tmp_path / "model.yaml"
        f.write_text(yaml.dump(desc))
        with pytest.raises(ValueError, match="nonexistent_layer"):
            load_model_description(str(f))

    def test_invalid_discrimination_method_raises(self, tmp_path):
        desc = self._base_desc()
        desc["optimization"] = {"discrimination_method": "magic"}
        f = tmp_path / "model.yaml"
        f.write_text(yaml.dump(desc))
        with pytest.raises(ValueError, match="discrimination_method"):
            load_model_description(str(f))

    def test_invalid_discrimination_mode_raises(self, tmp_path):
        desc = self._base_desc()
        desc["optimization"] = {"discrimination_mode": "destroy"}
        f = tmp_path / "model.yaml"
        f.write_text(yaml.dump(desc))
        with pytest.raises(ValueError, match="discrimination_mode"):
            load_model_description(str(f))


# ── build_alternate_experiments ──────────────────────────────────


class TestBuildAlternateExperiments:
    """Test building refl1d Experiments from alternate model specs."""

    @pytest.fixture()
    def q_dq(self):
        q = np.logspace(np.log10(0.008), np.log10(0.2), 30)
        return q, 0.025 * q

    def _three_layer_desc(self):
        return {
            "layers": [
                {"name": "air", "rho": 0},
                {
                    "name": "oxide",
                    "rho": 5.0,
                    "thickness": 20,
                    "interface": 3,
                    "fit": {"thickness": [5, 50]},
                },
                {"name": "Cu", "rho": 6.3},
            ],
        }

    def test_no_alternates_returns_empty(self, q_dq):
        q, dq = q_dq
        desc = self._three_layer_desc()
        result = build_alternate_experiments(desc, q, dq)
        assert result == []

    def test_remove_builds_experiment(self, q_dq):
        q, dq = q_dq
        desc = self._three_layer_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "no_oxide",
                    "modifications": [{"action": "remove", "layer": "oxide"}],
                }
            ]
        }
        alts = build_alternate_experiments(desc, q, dq)
        assert len(alts) == 1
        name, expt = alts[0]
        assert name == "no_oxide"
        _, r = expt.reflectivity()
        assert len(r) == len(q)

    def test_modify_changes_value(self, q_dq):
        q, dq = q_dq
        desc = self._three_layer_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "thin_oxide",
                    "modifications": [
                        {
                            "action": "modify",
                            "layer": "oxide",
                            "set": {"thickness": 5},
                        }
                    ],
                }
            ]
        }
        alts = build_alternate_experiments(desc, q, dq)
        assert len(alts) == 1
        name, expt = alts[0]
        assert name == "thin_oxide"
        _, r = expt.reflectivity()
        assert np.all(np.isfinite(r))

    def test_primary_unchanged_after_build(self, q_dq):
        """Ensure deep-copy means primary desc is not mutated."""
        q, dq = q_dq
        desc = self._three_layer_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "no_oxide",
                    "modifications": [{"action": "remove", "layer": "oxide"}],
                }
            ]
        }
        original_layer_count = len(desc["layers"])
        build_alternate_experiments(desc, q, dq)
        assert len(desc["layers"]) == original_layer_count

    def test_multiple_alternates(self, q_dq):
        q, dq = q_dq
        desc = self._three_layer_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "no_oxide",
                    "modifications": [{"action": "remove", "layer": "oxide"}],
                },
                {
                    "name": "thick_oxide",
                    "modifications": [
                        {
                            "action": "modify",
                            "layer": "oxide",
                            "set": {"thickness": 100},
                        }
                    ],
                },
            ]
        }
        alts = build_alternate_experiments(desc, q, dq)
        assert len(alts) == 2
        assert alts[0][0] == "no_oxide"
        assert alts[1][0] == "thick_oxide"


# ── build_alternate_descriptions ─────────────────────────────────


class TestBuildAlternateDescriptions:
    """Test building alternate description dicts without building experiments."""

    def _three_layer_desc(self):
        return {
            "layers": [
                {"name": "air", "rho": 0},
                {"name": "oxide", "rho": 5.0, "thickness": 20, "interface": 3},
                {"name": "Cu", "rho": 6.3},
            ],
        }

    def test_no_alternates_returns_empty(self):
        desc = self._three_layer_desc()
        result = build_alternate_descriptions(desc)
        assert result == []

    def test_remove_produces_fewer_layers(self):
        desc = self._three_layer_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "no_oxide",
                    "modifications": [{"action": "remove", "layer": "oxide"}],
                }
            ]
        }
        alt_descs = build_alternate_descriptions(desc)
        assert len(alt_descs) == 1
        name, alt = alt_descs[0]
        assert name == "no_oxide"
        assert len(alt["layers"]) == 2
        assert all(l["name"] != "oxide" for l in alt["layers"])

    def test_optimization_removed_from_alt(self):
        desc = self._three_layer_desc()
        desc["optimization"] = {
            "alternate_models": [
                {
                    "name": "no_oxide",
                    "modifications": [{"action": "remove", "layer": "oxide"}],
                }
            ]
        }
        _, alt = build_alternate_descriptions(desc)[0]
        assert "optimization" not in alt
