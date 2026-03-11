"""
Instrument simulation for synthetic reflectometry data.

Generates realistic noisy reflectivity curves by adding
Gaussian noise based on instrumental error characteristics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Maximum allowed data file size (50 MB).
MAX_DATA_FILE_SIZE = 50 * 1024 * 1024


class InstrumentSimulator:
    """Simulate instrumental noise on reflectivity data.

    Can be initialised from a measured data file (to inherit the
    real Q-grid and error bars) or with explicit Q-values and
    resolution parameters.

    Args:
        data_file: Path to a 4-column data file (Q, R, dR, dQ).
        q_values: Explicit Q-values array.
        dq_values: Resolution (scalar or array). Defaults to 0.025.
        relative_error: Fractional error when not derived from data.
    """

    def __init__(
        self,
        data_file: str | None = None,
        q_values: np.ndarray | None = None,
        dq_values: np.ndarray | float | None = 0.025,
        relative_error: float = 0.10,
    ):
        if data_file:
            data = load_measurement(data_file)
            self.q_values: np.ndarray = data["q"]
            self.dq_values: np.ndarray = data["dq"]
            self.relative_errors: np.ndarray = np.where(
                data["R"] == 0, relative_error, data["dR"] / data["R"]
            )
            self.relative_errors[self.relative_errors <= 0] = relative_error
        elif q_values is not None:
            self.q_values = np.asarray(q_values)
            n = len(self.q_values)
            if isinstance(dq_values, np.ndarray):
                if len(dq_values) != n:
                    raise ValueError("dq_values array must match length of q_values")
                self.dq_values = dq_values
            else:
                self.dq_values = float(dq_values) * np.ones(n)
            self.relative_errors = relative_error * np.ones(n)
        else:
            # Sensible defaults for a typical reflectometer
            self.q_values = np.logspace(np.log10(0.008), np.log10(0.2), 50)
            n = len(self.q_values)
            if isinstance(dq_values, np.ndarray):
                if len(dq_values) != n:
                    raise ValueError("dq_values array must match length of q_values")
                self.dq_values = dq_values
            else:
                self.dq_values = float(dq_values) * np.ones(n)
            self.relative_errors = relative_error * np.ones(n)

        if not (len(self.q_values) == len(self.dq_values) == len(self.relative_errors)):
            raise ValueError(
                "q_values, dq_values, and relative_errors must have the same length"
            )

    def add_noise(
        self,
        reflectivity: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add Gaussian noise to a reflectivity curve.

        Args:
            reflectivity: Clean reflectivity values.
            rng: Explicit NumPy random generator.  When ``None`` a new
                 unseeded generator is created.  Pass an explicit
                 generator for reproducibility and parallel safety.

        Returns:
            Tuple of (noisy_reflectivity, error_bars).
        """
        if rng is None:
            rng = np.random.default_rng()
        errors = self.relative_errors * reflectivity
        noise = rng.normal(0, errors)
        noisy_reflectivity = reflectivity + noise
        return noisy_reflectivity, errors


def load_measurement(filename: str) -> dict[str, np.ndarray]:
    """Load measurement data from a 4-column text file.

    Expects columns: Q, R, dR, dQ.

    Args:
        filename: Path to the data file.

    Returns:
        Dict with keys ``q``, ``R``, ``dR``, ``dq``.

    Raises:
        ValueError: If the file has fewer than 4 columns.
    """
    file_size = Path(filename).stat().st_size
    if file_size > MAX_DATA_FILE_SIZE:
        raise ValueError(
            f"Data file too large ({file_size} bytes); "
            f"max is {MAX_DATA_FILE_SIZE} bytes"
        )
    data = np.loadtxt(filename)
    if data.shape[1] < 4:
        raise ValueError("Data file must have at least 4 columns: Q, R, dR, dQ")
    return {
        "q": data[:, 0],
        "R": data[:, 1],
        "dR": data[:, 2],
        "dq": data[:, 3],
    }
