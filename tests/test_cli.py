"""
Tests for the ROSE CLI (click command group).
"""

from click.testing import CliRunner

from rose.cli import main


def test_cli_help():
    """Top-level --help exits cleanly."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "ROSE" in result.output


def test_cli_version():
    """--version prints the package version."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_inspect_help():
    """rose inspect --help exits cleanly."""
    runner = CliRunner()
    result = runner.invoke(main, ["inspect", "--help"])

    assert result.exit_code == 0
    assert "MODEL_FILE" in result.output


def test_optimize_help():
    """rose optimize --help exits cleanly."""
    runner = CliRunner()
    result = runner.invoke(main, ["optimize", "--help"])

    assert result.exit_code == 0
    assert "--model-file" in result.output
    assert "--param" in result.output


def test_inspect_example_model():
    """rose inspect actually loads a model and lists params."""
    runner = CliRunner()
    result = runner.invoke(main, ["inspect", "examples/models/layer_a_on_b.yaml"])

    assert result.exit_code == 0
    assert "layer_A thickness" in result.output
    assert "Variable" in result.output


def test_inspect_verbose_shows_fixed():
    """rose inspect --verbose shows fixed params."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["inspect", "--verbose", "examples/models/layer_a_on_b.yaml"]
    )

    assert result.exit_code == 0
    assert "Fixed" in result.output


def test_report_help():
    """rose report --help exits cleanly."""
    runner = CliRunner()
    result = runner.invoke(main, ["report", "--help"])

    assert result.exit_code == 0
    assert "--result-file" in result.output
