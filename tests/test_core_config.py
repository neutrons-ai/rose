"""Tests for rose.core.config."""

import os
import tempfile
from pathlib import Path

import yaml

from rose.core.config import MCMCSettings, RoseConfig, load_config


def test_default_config():
    """Test default configuration values."""
    config = load_config()
    assert config.mcmc.steps == 2000
    assert config.mcmc.entropy_method == "kdn"
    assert config.instrument.q_min == 0.008
    assert config.num_realizations == 3
    assert config.parallel is True


def test_load_config_from_yaml():
    """Test loading config from a YAML file."""
    data = {
        "mcmc": {"steps": 500, "entropy_method": "mvn"},
        "num_realizations": 5,
        "output_dir": "/tmp/custom",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        config = load_config(f.name)

    os.unlink(f.name)

    assert config.mcmc.steps == 500
    assert config.mcmc.entropy_method == "mvn"
    assert config.num_realizations == 5
    assert config.output_dir == "/tmp/custom"
    # Unchanged defaults
    assert config.mcmc.burn == 1000
    assert config.instrument.q_max == 0.20


def test_load_config_file_not_found():
    """Test that missing config file raises FileNotFoundError."""
    try:
        load_config("/nonexistent/path.yaml")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_env_override(monkeypatch):
    """Test that ROSE_OUTPUT_DIR env var overrides config."""
    monkeypatch.setenv("ROSE_OUTPUT_DIR", "/env/results")
    config = load_config()
    assert config.output_dir == "/env/results"
