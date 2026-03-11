"""
Data access layer for ROSE optimization results.

Provides :class:`ResultData` which loads ``optimization_results.json``
files produced by ``rose optimize``.

Typical result directory layout::

    results/
    ├── optimization_results.json
    ├── information_gain.png
    ├── reflectivity_*.png
    └── sld_profile_*.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultData:
    """Read-only accessor for a single optimization result directory.

    Args:
        result_dir: Path to a directory containing
            ``optimization_results.json``.
    """

    def __init__(self, result_dir: str | Path) -> None:
        self._dir = Path(result_dir)
        self._data: dict[str, Any] | None = None

    @property
    def path(self) -> Path:
        return self._dir

    @property
    def name(self) -> str:
        return self._dir.name

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            result_file = self._dir / "optimization_results.json"
            if not result_file.exists():
                raise FileNotFoundError(f"No optimization_results.json in {self._dir}")
            self._data = json.loads(result_file.read_text())
        return self._data

    def exists(self) -> bool:
        """Return ``True`` if the result file exists."""
        return (self._dir / "optimization_results.json").exists()

    # ── Summary ──────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Return a summary suitable for listing cards.

        Returns:
            Dict with keys: name, parameter, optimal_value,
            max_information_gain, num_values, num_realizations,
            settings.
        """
        d = self._load()
        settings = d.get("settings", {})
        return {
            "name": self.name,
            "parameter": d.get("parameter", "unknown"),
            "optimal_value": d.get("optimal_value"),
            "max_information_gain": d.get("max_information_gain"),
            "max_information_gain_std": d.get("max_information_gain_std"),
            "prior_entropy": d.get("prior_entropy"),
            "num_values": len(d.get("parameter_values", [])),
            "num_realizations": len(d.get("simulated_data", [[]])[0])
            if d.get("simulated_data")
            else 0,
            "settings": settings,
        }

    # ── Information gain curve ───────────────────────────────

    def get_info_gain(self) -> dict[str, Any]:
        """Return data for the information gain curve.

        Returns:
            Dict with keys: parameter, values, info_gain,
            info_gain_std, optimal_value, max_information_gain.
        """
        d = self._load()
        values = []
        info_gain = []
        info_gain_std = []
        for row in d.get("results", []):
            values.append(row[0])
            info_gain.append(row[1])
            info_gain_std.append(row[2])

        return {
            "parameter": d.get("parameter", ""),
            "values": values,
            "info_gain": info_gain,
            "info_gain_std": info_gain_std,
            "optimal_value": d.get("optimal_value"),
            "max_information_gain": d.get("max_information_gain"),
        }

    # ── Reflectivity data ────────────────────────────────────

    def get_reflectivity(self, param_index: int = 0) -> dict[str, Any]:
        """Return reflectivity curves for a given parameter value index.

        Args:
            param_index: Index into parameter_values list.

        Returns:
            Dict with keys: param_value, realizations (list of
            dicts with q, reflectivity, noisy_reflectivity, errors).
        """
        d = self._load()
        param_values = d.get("parameter_values", [])
        sim_data = d.get("simulated_data", [])

        if param_index < 0 or param_index >= len(sim_data):
            return {"param_value": None, "realizations": []}

        realizations = []
        for r in sim_data[param_index]:
            realizations.append(
                {
                    "q": r.get("q_values", []),
                    "reflectivity": r.get("reflectivity", []),
                    "noisy_reflectivity": r.get("noisy_reflectivity", []),
                    "errors": r.get("errors", []),
                }
            )

        return {
            "param_value": param_values[param_index]
            if param_index < len(param_values)
            else None,
            "realizations": realizations,
        }

    # ── SLD profiles ─────────────────────────────────────────

    def get_sld(self, param_index: int = 0) -> dict[str, Any]:
        """Return SLD profiles for a given parameter value index.

        Args:
            param_index: Index into parameter_values list.

        Returns:
            Dict with keys: param_value, realizations (list of
            dicts with z, sld_best, sld_low, sld_high).
        """
        d = self._load()
        param_values = d.get("parameter_values", [])
        sim_data = d.get("simulated_data", [])

        if param_index < 0 or param_index >= len(sim_data):
            return {"param_value": None, "realizations": []}

        realizations = []
        for r in sim_data[param_index]:
            realizations.append(
                {
                    "z": r.get("z", []),
                    "sld_best": r.get("sld_best", []),
                    "sld_low": r.get("sld_low", []),
                    "sld_high": r.get("sld_high", []),
                }
            )

        return {
            "param_value": param_values[param_index]
            if param_index < len(param_values)
            else None,
            "realizations": realizations,
        }

    # ── Settings / model info ────────────────────────────────

    def get_settings(self) -> dict[str, Any]:
        """Return the optimization settings."""
        d = self._load()
        return d.get("settings", {})

    def get_model_yaml(self) -> str | None:
        """Return the model YAML source if a .yaml file exists."""
        for p in sorted(self._dir.glob("*.yaml")):
            return p.read_text()
        for p in sorted(self._dir.glob("*.yml")):
            return p.read_text()
        return None


def list_results(results_dir: str | Path) -> list[ResultData]:
    """List all result directories under *results_dir*.

    A directory is considered a result if it contains
    ``optimization_results.json``.

    Returns:
        Sorted list of :class:`ResultData` instances.
    """
    base = Path(results_dir)
    if not base.is_dir():
        return []

    results: list[ResultData] = []

    # Check if the base dir itself has results
    if (base / "optimization_results.json").exists():
        results.append(ResultData(base))

    # Check subdirectories (one level deep)
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "optimization_results.json").exists():
            results.append(ResultData(child))

    return results
