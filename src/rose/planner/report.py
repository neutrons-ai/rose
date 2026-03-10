"""
Report generation for optimization results.

Produces matplotlib plots of information gain curves,
simulated reflectivity data, and SLD depth profiles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

logger = logging.getLogger(__name__)

#: Maximum allowed JSON result file size (100 MB).
MAX_JSON_FILE_SIZE = 100 * 1024 * 1024
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt  # noqa: E402


def make_report(
    json_file: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Generate plots from an optimization results JSON file.

    Creates:
      - ``information_gain.png`` — ΔH vs. parameter value
      - ``simulated_data_<i>.png`` — reflectivity per parameter value
      - ``sld_contours_<i>.png`` — SLD profiles with confidence bands

    Args:
        json_file: Path to ``optimization_results.json``.
        output_dir: Directory to write plots into (created if needed).

    Returns:
        List of paths to generated image files.
    """
    json_path = Path(json_file)
    file_size = json_path.stat().st_size
    if file_size > MAX_JSON_FILE_SIZE:
        raise ValueError(
            f"Result file too large ({file_size} bytes); "
            f"max is {MAX_JSON_FILE_SIZE} bytes"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        result_dict = json.load(f)

    results = result_dict["results"]
    simulated_data = result_dict["simulated_data"]

    generated: list[str] = []

    # --- Information gain curve ---
    param_values = [r[0] for r in results]
    info_gains = [r[1] for r in results]
    std_gains = [r[2] for r in results] if len(results[0]) > 2 else None

    fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.17)
    if std_gains:
        ax.errorbar(param_values, info_gains, yerr=std_gains, marker="o", capsize=3)
    else:
        ax.plot(param_values, info_gains, marker="o")
    ax.set_xlabel("Parameter Value", fontsize=13)
    ax.set_ylabel("Information Gain (bits)", fontsize=13)
    ax.set_title(f"Optimization: {result_dict.get('parameter', '')}", fontsize=13)
    ax.grid(True, alpha=0.3)

    path = str(output_dir / "information_gain.png")
    fig.savefig(path)
    plt.close(fig)
    generated.append(path)

    # --- Per-value reflectivity + SLD plots ---
    for i, real_set in enumerate(simulated_data):
        # Reflectivity
        fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.17)
        for j, data in enumerate(real_set):
            ax.errorbar(
                data["q_values"],
                data["noisy_reflectivity"],
                yerr=data["errors"],
                linewidth=1,
                markersize=2,
                marker=".",
                linestyle="",
                label=f"Realization {j + 1}",
            )
            if j == 0:
                ax.plot(
                    data["q_values"],
                    data["reflectivity"],
                    label="Best fit",
                    color="black",
                    linewidth=1.5,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Q ($\AA^{-1}$)", fontsize=13)
        ax.set_ylabel("Reflectivity", fontsize=13)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"Value = {param_values[i]:.3f}", fontsize=13)

        path = str(output_dir / f"simulated_data_{i}.png")
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)

        # SLD contours
        fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.17)
        for j, data in enumerate(real_set):
            if not data.get("sld_best"):
                continue
            sld_best = np.asarray(data["sld_best"])
            z = np.asarray(data["z"])

            # Find the interface start and shift z
            start_idx = len(sld_best) - 1
            for k in range(len(sld_best) - 1, 0, -1):
                if abs(sld_best[k] - sld_best[k - 1]) > 0.001:
                    start_idx = k
                    break
            shifted_z = z - z[start_idx]

            ax.plot(
                shifted_z[:start_idx],
                sld_best[:start_idx],
                linewidth=2,
                label=f"Realization {j + 1}",
            )
            ax.fill_between(
                shifted_z[:start_idx],
                np.asarray(data["sld_low"])[:start_idx],
                np.asarray(data["sld_high"])[:start_idx],
                alpha=0.2,
                color=ax.lines[-1].get_color(),
            )

        ax.set_xlabel(r"z ($\AA$)", fontsize=13)
        ax.set_ylabel(r"SLD ($10^{-6}/\AA^2$)", fontsize=13)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"Value = {param_values[i]:.3f}", fontsize=13)

        path = str(output_dir / f"sld_contours_{i}.png")
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)

    return generated
