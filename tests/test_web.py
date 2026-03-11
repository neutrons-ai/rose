"""Tests for the ROSE web application (Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Fixtures ─────────────────────────────────────────────────────

SAMPLE_RESULT = {
    "parameter": "layer_b thickness",
    "parameter_values": [10.0, 20.0, 30.0],
    "results": [
        [10.0, 3.5, 0.5],
        [20.0, 5.2, 0.8],
        [30.0, 4.1, 0.6],
    ],
    "optimal_value": 20.0,
    "max_information_gain": 5.2,
    "max_information_gain_std": 0.8,
    "prior_entropy": 8.0,
    "settings": {
        "entropy_method": "kdn",
        "mcmc_steps": 1000,
        "num_realizations": 2,
    },
    "simulated_data": [
        # For each parameter value, a list of realizations
        [
            {
                "q_values": [0.01, 0.05, 0.1],
                "dq_values": [0.001, 0.005, 0.01],
                "reflectivity": [1.0, 0.5, 0.1],
                "noisy_reflectivity": [1.02, 0.48, 0.11],
                "errors": [0.02, 0.03, 0.01],
                "z": [0.0, 10.0, 20.0, 30.0],
                "sld_best": [0.0, 2.0, 4.5, 2.07],
                "sld_low": [0.0, 1.5, 4.0, 1.9],
                "sld_high": [0.0, 2.5, 5.0, 2.2],
                "posterior_entropy": 5.0,
            },
            {
                "q_values": [0.01, 0.05, 0.1],
                "dq_values": [0.001, 0.005, 0.01],
                "reflectivity": [1.0, 0.52, 0.09],
                "noisy_reflectivity": [0.99, 0.54, 0.10],
                "errors": [0.02, 0.03, 0.01],
                "z": [0.0, 10.0, 20.0, 30.0],
                "sld_best": [0.0, 2.1, 4.6, 2.07],
                "sld_low": [0.0, 1.6, 4.1, 1.9],
                "sld_high": [0.0, 2.6, 5.1, 2.2],
                "posterior_entropy": 4.8,
            },
        ],
        [
            {
                "q_values": [0.01, 0.05, 0.1],
                "dq_values": [0.001, 0.005, 0.01],
                "reflectivity": [1.0, 0.6, 0.15],
                "noisy_reflectivity": [1.01, 0.59, 0.16],
                "errors": [0.02, 0.03, 0.01],
                "z": [0.0, 10.0, 20.0, 30.0],
                "sld_best": [0.0, 2.0, 5.0, 2.07],
                "sld_low": [0.0, 1.5, 4.5, 1.9],
                "sld_high": [0.0, 2.5, 5.5, 2.2],
                "posterior_entropy": 3.5,
            },
            {
                "q_values": [0.01, 0.05, 0.1],
                "dq_values": [0.001, 0.005, 0.01],
                "reflectivity": [1.0, 0.58, 0.14],
                "noisy_reflectivity": [0.98, 0.60, 0.13],
                "errors": [0.02, 0.03, 0.01],
                "z": [0.0, 10.0, 20.0, 30.0],
                "sld_best": [0.0, 2.1, 5.1, 2.07],
                "sld_low": [0.0, 1.6, 4.6, 1.9],
                "sld_high": [0.0, 2.6, 5.6, 2.2],
                "posterior_entropy": 3.3,
            },
        ],
        [
            {
                "q_values": [0.01, 0.05, 0.1],
                "dq_values": [0.001, 0.005, 0.01],
                "reflectivity": [1.0, 0.55, 0.12],
                "noisy_reflectivity": [1.01, 0.53, 0.11],
                "errors": [0.02, 0.03, 0.01],
                "z": [0.0, 10.0, 20.0, 30.0],
                "sld_best": [0.0, 2.0, 4.8, 2.07],
                "sld_low": [0.0, 1.5, 4.3, 1.9],
                "sld_high": [0.0, 2.5, 5.3, 2.2],
                "posterior_entropy": 4.0,
            },
            {
                "q_values": [0.01, 0.05, 0.1],
                "dq_values": [0.001, 0.005, 0.01],
                "reflectivity": [1.0, 0.56, 0.11],
                "noisy_reflectivity": [0.99, 0.57, 0.12],
                "errors": [0.02, 0.03, 0.01],
                "z": [0.0, 10.0, 20.0, 30.0],
                "sld_best": [0.0, 2.1, 4.9, 2.07],
                "sld_low": [0.0, 1.6, 4.4, 1.9],
                "sld_high": [0.0, 2.6, 5.4, 2.2],
                "posterior_entropy": 3.9,
            },
        ],
    ],
}


@pytest.fixture()
def results_dir(tmp_path: Path):
    """Create a temporary results directory with sample data."""
    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    (run_dir / "optimization_results.json").write_text(json.dumps(SAMPLE_RESULT))
    # Also add a model YAML file
    (run_dir / "generated_model.yaml").write_text(
        "name: test model\nlayers:\n  - name: air\n    rho: 0\n"
    )
    return tmp_path


@pytest.fixture()
def app(results_dir):
    """Create a Flask test app."""
    from rose.web import create_app

    return create_app(str(results_dir))


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()


# ── Page route tests ─────────────────────────────────────────────


class TestPageRoutes:
    """Tests for HTML page routes."""

    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_shows_result(self, client):
        resp = client.get("/")
        assert b"run_01" in resp.data
        assert b"layer_b thickness" in resp.data

    def test_result_detail_returns_200(self, client):
        resp = client.get("/results/run_01")
        assert resp.status_code == 200

    def test_result_detail_shows_parameter(self, client):
        resp = client.get("/results/run_01")
        assert b"layer_b thickness" in resp.data

    def test_result_detail_not_found(self, client):
        resp = client.get("/results/nonexistent")
        assert resp.status_code == 404

    def test_model_view_returns_200(self, client):
        resp = client.get("/results/run_01/model")
        assert resp.status_code == 200
        assert b"test model" in resp.data

    def test_model_view_not_found(self, client):
        resp = client.get("/results/nonexistent/model")
        assert resp.status_code == 404

    def test_path_traversal_rejected(self, client):
        resp = client.get("/results/../etc/passwd")
        assert resp.status_code == 404


# ── JSON API tests ───────────────────────────────────────────────


class TestAPIRoutes:
    """Tests for JSON API endpoints."""

    def test_api_results_list(self, client):
        resp = client.get("/api/results")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["parameter"] == "layer_b thickness"
        assert data[0]["name"] == "run_01"

    def test_api_info_gain(self, client):
        resp = client.get("/api/results/run_01/info-gain")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["parameter"] == "layer_b thickness"
        assert data["values"] == [10.0, 20.0, 30.0]
        assert len(data["info_gain"]) == 3
        assert data["optimal_value"] == 20.0

    def test_api_reflectivity(self, client):
        resp = client.get("/api/results/run_01/reflectivity?index=0")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["param_value"] == 10.0
        assert len(data["realizations"]) == 2
        r0 = data["realizations"][0]
        assert "q" in r0
        assert "reflectivity" in r0
        assert "noisy_reflectivity" in r0

    def test_api_reflectivity_default_index(self, client):
        resp = client.get("/api/results/run_01/reflectivity")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["param_value"] == 10.0

    def test_api_reflectivity_out_of_range(self, client):
        resp = client.get("/api/results/run_01/reflectivity?index=99")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["param_value"] is None
        assert data["realizations"] == []

    def test_api_sld(self, client):
        resp = client.get("/api/results/run_01/sld?index=1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["param_value"] == 20.0
        assert len(data["realizations"]) == 2
        r0 = data["realizations"][0]
        assert "z" in r0
        assert "sld_best" in r0
        assert "sld_low" in r0
        assert "sld_high" in r0

    def test_api_settings(self, client):
        resp = client.get("/api/results/run_01/settings")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["entropy_method"] == "kdn"
        assert data["mcmc_steps"] == 1000

    def test_api_summary(self, client):
        resp = client.get("/api/results/run_01/summary")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["parameter"] == "layer_b thickness"
        assert data["num_values"] == 3
        assert data["num_realizations"] == 2

    def test_api_not_found(self, client):
        resp = client.get("/api/results/bad_id/info-gain")
        assert resp.status_code == 404


# ── Data layer tests ─────────────────────────────────────────────


class TestResultData:
    """Tests for the ResultData class."""

    def test_exists_true(self, results_dir):
        from rose.web.data import ResultData

        rd = ResultData(results_dir / "run_01")
        assert rd.exists()

    def test_exists_false(self, tmp_path):
        from rose.web.data import ResultData

        rd = ResultData(tmp_path / "nonexistent")
        assert not rd.exists()

    def test_get_summary(self, results_dir):
        from rose.web.data import ResultData

        rd = ResultData(results_dir / "run_01")
        s = rd.get_summary()
        assert s["parameter"] == "layer_b thickness"
        assert s["optimal_value"] == 20.0
        assert s["num_values"] == 3

    def test_get_info_gain(self, results_dir):
        from rose.web.data import ResultData

        rd = ResultData(results_dir / "run_01")
        ig = rd.get_info_gain()
        assert ig["values"] == [10.0, 20.0, 30.0]
        assert len(ig["info_gain"]) == 3

    def test_get_model_yaml(self, results_dir):
        from rose.web.data import ResultData

        rd = ResultData(results_dir / "run_01")
        yaml = rd.get_model_yaml()
        assert yaml is not None
        assert "test model" in yaml

    def test_get_model_yaml_missing(self, tmp_path):
        from rose.web.data import ResultData

        run_dir = tmp_path / "no_model"
        run_dir.mkdir()
        (run_dir / "optimization_results.json").write_text(json.dumps(SAMPLE_RESULT))
        rd = ResultData(run_dir)
        assert rd.get_model_yaml() is None

    def test_list_results(self, results_dir):
        from rose.web.data import list_results

        items = list_results(results_dir)
        assert len(items) == 1
        assert items[0].name == "run_01"

    def test_list_results_empty(self, tmp_path):
        from rose.web.data import list_results

        items = list_results(tmp_path)
        assert items == []

    def test_list_results_nonexistent(self, tmp_path):
        from rose.web.data import list_results

        items = list_results(tmp_path / "nonexistent")
        assert items == []


# ── CLI tests ────────────────────────────────────────────────────


class TestServeCLI:
    """Tests for the rose serve CLI command."""

    def test_serve_help(self):
        from click.testing import CliRunner

        from rose.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "RESULTS_DIR" in result.output
        assert "--port" in result.output
        assert "--no-browser" in result.output
