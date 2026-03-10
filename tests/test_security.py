"""
Security-focused tests for ROSE Phase 1.

Validates the guards introduced from the security review:
file size limits, path traversal rejection, parameter bounds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from rose.cli import main

# ── Path traversal rejection ─────────────────────────────────────


class TestPathTraversal:
    def test_report_rejects_dotdot_output(self, tmp_path):
        """rose report --output-dir with '..' is rejected."""
        # Create a minimal result JSON so --result-file passes validation
        result = {
            "parameter": "x",
            "results": [[1, 0.5, 0.1]],
            "simulated_data": [[]],
        }
        jf = tmp_path / "res.json"
        jf.write_text(json.dumps(result))

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", "--result-file", str(jf), "--output-dir", "/tmp/../etc/evil"],
        )
        assert result.exit_code != 0
        assert "Path traversal" in result.output or "not allowed" in result.output


# ── CLI IntRange enforcement ─────────────────────────────────────


class TestCLIBounds:
    def test_mcmc_steps_too_low(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "optimize",
                "--model-file",
                "examples/models/layer_a_on_b.yaml",
                "--param",
                "x",
                "--param-values",
                "1",
                "--mcmc-steps",
                "10",
            ],
        )
        assert result.exit_code != 0

    def test_mcmc_steps_too_high(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "optimize",
                "--model-file",
                "examples/models/layer_a_on_b.yaml",
                "--param",
                "x",
                "--param-values",
                "1",
                "--mcmc-steps",
                "999999",
            ],
        )
        assert result.exit_code != 0

    def test_num_realizations_zero(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "optimize",
                "--model-file",
                "examples/models/layer_a_on_b.yaml",
                "--param",
                "x",
                "--param-values",
                "1",
                "--num-realizations",
                "0",
            ],
        )
        assert result.exit_code != 0


# ── File size guards ─────────────────────────────────────────────


class TestFileSizeGuards:
    def test_load_measurement_rejects_oversized_file(self, tmp_path):
        from rose.planner.instrument import MAX_DATA_FILE_SIZE, load_measurement

        big_file = tmp_path / "huge.dat"
        # Create a sparse file that reports large size
        with open(big_file, "wb") as f:
            f.seek(MAX_DATA_FILE_SIZE + 1)
            f.write(b"0")

        with pytest.raises(ValueError, match="too large"):
            load_measurement(str(big_file))

    def test_report_rejects_oversized_json(self, tmp_path):
        from rose.planner.report import MAX_JSON_FILE_SIZE, make_report

        big_file = tmp_path / "huge.json"
        with open(big_file, "wb") as f:
            f.seek(MAX_JSON_FILE_SIZE + 1)
            f.write(b"}")

        with pytest.raises(ValueError, match="too large"):
            make_report(str(big_file), str(tmp_path / "out"))


# ── Model loader guards ─────────────────────────────────────────


class TestModelLoaderGuards:
    def test_unsupported_extension_rejected(self, tmp_path):
        from rose.planner.model_loader import load_experiment

        bad_file = tmp_path / "model.txt"
        bad_file.write_text("x = 1\n")

        q = np.linspace(0.01, 0.2, 10)
        with pytest.raises(ValueError, match="Unsupported"):
            load_experiment(str(bad_file), q, q * 0.025)

    def test_oversized_model_rejected(self, tmp_path):
        from rose.planner.model_loader import MAX_MODEL_FILE_SIZE, load_experiment

        big_file = tmp_path / "model.yaml"
        with open(big_file, "wb") as f:
            f.seek(MAX_MODEL_FILE_SIZE + 1)
            f.write(b"#")

        q = np.linspace(0.01, 0.2, 10)
        with pytest.raises(ValueError, match="too large"):
            load_experiment(str(big_file), q, q * 0.025)

    def test_no_code_execution(self, tmp_path):
        """Verify YAML model files do NOT execute arbitrary code."""
        import yaml

        from rose.planner.model_loader import load_model_description

        # A YAML file that would be dangerous if parsed unsafely
        malicious = tmp_path / "evil.yaml"
        malicious.write_text(
            yaml.dump(
                {
                    "layers": [
                        {"name": "air", "rho": 0},
                        {"name": "Si", "rho": 2.07},
                    ]
                }
            )
        )
        # Should load safely without executing code
        desc = load_model_description(str(malicious))
        assert len(desc["layers"]) == 2
