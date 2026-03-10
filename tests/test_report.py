"""
Tests for rose.planner.report.

Generates a minimal results JSON and verifies that make_report
produces the expected plot files.
"""

from __future__ import annotations

import json

import pytest

from rose.planner.report import make_report


@pytest.fixture()
def result_json(tmp_path):
    """Create a minimal optimization_results.json."""
    data = {
        "parameter": "test_param",
        "parameter_values": [10.0, 20.0, 30.0],
        "results": [
            [10.0, 1.5, 0.2],
            [20.0, 2.3, 0.3],
            [30.0, 1.8, 0.1],
        ],
        "simulated_data": [
            [
                {
                    "q_values": [0.01, 0.02, 0.03],
                    "reflectivity": [0.9, 0.5, 0.1],
                    "noisy_reflectivity": [0.91, 0.48, 0.12],
                    "errors": [0.01, 0.01, 0.01],
                    "z": [0, 10, 20, 30],
                    "sld_best": [0.0, 3.0, 4.5, 2.07],
                    "sld_low": [0.0, 2.5, 4.0, 2.0],
                    "sld_high": [0.0, 3.5, 5.0, 2.1],
                    "posterior_entropy": 4.5,
                }
            ],
            [
                {
                    "q_values": [0.01, 0.02, 0.03],
                    "reflectivity": [0.85, 0.45, 0.08],
                    "noisy_reflectivity": [0.86, 0.44, 0.09],
                    "errors": [0.01, 0.01, 0.01],
                    "z": [0, 10, 20, 30],
                    "sld_best": [0.0, 3.0, 4.5, 2.07],
                    "sld_low": [0.0, 2.5, 4.0, 2.0],
                    "sld_high": [0.0, 3.5, 5.0, 2.1],
                    "posterior_entropy": 4.2,
                }
            ],
            [
                {
                    "q_values": [0.01, 0.02, 0.03],
                    "reflectivity": [0.8, 0.4, 0.06],
                    "noisy_reflectivity": [0.82, 0.39, 0.07],
                    "errors": [0.01, 0.01, 0.01],
                    "z": [0, 10, 20, 30],
                    "sld_best": [0.0, 3.0, 4.5, 2.07],
                    "sld_low": [0.0, 2.5, 4.0, 2.0],
                    "sld_high": [0.0, 3.5, 5.0, 2.1],
                    "posterior_entropy": 4.0,
                }
            ],
        ],
    }
    fpath = tmp_path / "optimization_results.json"
    with open(fpath, "w") as f:
        json.dump(data, f)
    return str(fpath)


def test_make_report_creates_info_gain_png(result_json, tmp_path):
    output_dir = tmp_path / "plots"
    paths = make_report(result_json, str(output_dir))
    assert any("information_gain.png" in p for p in paths)


def test_make_report_creates_simulated_data_pngs(result_json, tmp_path):
    output_dir = tmp_path / "plots"
    paths = make_report(result_json, str(output_dir))
    sim_plots = [p for p in paths if "simulated_data" in p]
    # One per parameter value (3 values)
    assert len(sim_plots) == 3


def test_make_report_creates_sld_contour_pngs(result_json, tmp_path):
    output_dir = tmp_path / "plots"
    paths = make_report(result_json, str(output_dir))
    sld_plots = [p for p in paths if "sld_contours" in p]
    assert len(sld_plots) == 3


def test_make_report_files_exist(result_json, tmp_path):
    output_dir = tmp_path / "plots"
    paths = make_report(result_json, str(output_dir))
    from pathlib import Path

    for p in paths:
        assert Path(p).exists()
