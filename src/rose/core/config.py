"""
Configuration loading for ROSE.

Supports YAML config files and environment variables
via python-dotenv. Sensible defaults are provided for
all settings so that the tool works out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MCMCSettings:
    """Settings for MCMC sampling.

    Attributes:
        steps: Number of MCMC steps after burn-in.
        burn: Number of burn-in steps.
        population: Population multiplier for DREAM sampler.
        entropy_method: Entropy calculation method ("mvn" or "kdn").
    """

    steps: int = 2000
    burn: int = 1000
    population: int = 10
    entropy_method: str = "kdn"


@dataclass
class InstrumentSettings:
    """Settings for the instrument simulator.

    Attributes:
        q_min: Minimum Q value (Å⁻¹).
        q_max: Maximum Q value (Å⁻¹).
        q_points: Number of Q points.
        dq_fraction: Fractional dQ resolution (dQ/Q).
        relative_error: Relative error for noise simulation.
    """

    q_min: float = 0.008
    q_max: float = 0.20
    q_points: int = 200
    dq_fraction: float = 0.025
    relative_error: float = 0.10


@dataclass
class RoseConfig:
    """Top-level configuration for ROSE.

    Attributes:
        mcmc: MCMC sampler settings.
        instrument: Instrument simulator settings.
        num_realizations: Default number of noise realizations.
        parallel: Whether to run optimizations in parallel.
        output_dir: Default output directory.
    """

    mcmc: MCMCSettings = field(default_factory=MCMCSettings)
    instrument: InstrumentSettings = field(default_factory=InstrumentSettings)
    num_realizations: int = 3
    parallel: bool = True
    output_dir: str = "results"


def load_config(config_path: str | Path | None = None) -> RoseConfig:
    """Load configuration from a YAML file.

    If no path is provided, returns default settings.
    Environment variables prefixed with ``ROSE_`` override
    YAML values (e.g. ``ROSE_OUTPUT_DIR``).

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Populated RoseConfig instance.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
    """
    config = RoseConfig()

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        if "mcmc" in data:
            for key, value in data["mcmc"].items():
                if hasattr(config.mcmc, key):
                    setattr(config.mcmc, key, value)

        if "instrument" in data:
            for key, value in data["instrument"].items():
                if hasattr(config.instrument, key):
                    setattr(config.instrument, key, value)

        for key in ("num_realizations", "parallel", "output_dir"):
            if key in data:
                setattr(config, key, data[key])

    # Environment variable overrides
    env_output = os.environ.get("ROSE_OUTPUT_DIR")
    if env_output is not None:
        config.output_dir = env_output

    return config
