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

    disc_data = result_dict.get("discrimination")
    disc_mode = disc_data.get("mode", "report") if disc_data else "report"
    eff_gains = None
    if disc_data and disc_mode == "penalize" and disc_data.get("per_value"):
        eff_gains = [
            pv.get("effective_info_gain", ig)
            for pv, ig in zip(disc_data["per_value"], info_gains)
        ]

    fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.17)
    if eff_gains is not None:
        # Show both raw (faded) and penalized (bold) on the same axes
        if std_gains:
            ax.errorbar(
                param_values,
                info_gains,
                yerr=std_gains,
                marker="o",
                capsize=3,
                color="tab:blue",
                alpha=0.35,
                label="Raw ΔH",
            )
        else:
            ax.plot(
                param_values,
                info_gains,
                marker="o",
                color="tab:blue",
                alpha=0.35,
                label="Raw ΔH",
            )
        ax.plot(
            param_values,
            eff_gains,
            marker="^",
            color="tab:green",
            linewidth=2,
            label="Penalized ΔH",
        )
        ax.legend(frameon=False, fontsize=9)
    else:
        if std_gains:
            ax.errorbar(
                param_values,
                info_gains,
                yerr=std_gains,
                marker="o",
                capsize=3,
            )
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

    # --- Model discrimination plot ---
    disc_data = result_dict.get("discrimination")
    if disc_data and disc_data.get("per_value"):
        disc_plots = _plot_discrimination(
            param_values, disc_data, info_gains, std_gains, output_dir
        )
        generated.extend(disc_plots)

    return generated


def _plot_discrimination(
    param_values: list[float],
    disc_data: dict,
    info_gains: list[float],
    std_gains: list[float] | None,
    output_dir: Path,
) -> list[str]:
    """Generate model discrimination plots.

    Creates:
      - ``model_discrimination.png`` — P(primary) per alternate model
        vs. parameter value, with info gain on a twin axis.

    Returns:
        List of generated file paths.
    """
    generated: list[str] = []
    per_value = disc_data["per_value"]
    alt_names = disc_data["alternate_models"]
    mode = disc_data.get("mode", "report")

    # --- P(primary) + info gain twin-axis plot ---
    fig, ax1 = plt.subplots(dpi=150, figsize=(7, 4.5))
    fig.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.17)

    # Plot P(primary | data) for each alternate on left axis
    for aname in alt_names:
        probs = [
            pv.get("mean_model_prob", {}).get(aname, float("nan")) for pv in per_value
        ]
        ax1.plot(param_values, probs, marker="s", label=f"P(primary) vs {aname}")

    ax1.set_xlabel("Parameter Value", fontsize=13)
    ax1.set_ylabel("P(primary | data)", fontsize=13)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Info gain on right twin axis
    ax2 = ax1.twinx()
    if std_gains:
        ax2.errorbar(
            param_values,
            info_gains,
            yerr=std_gains,
            marker="o",
            capsize=3,
            color="tab:red",
            alpha=0.6,
            label="ΔH",
        )
    else:
        ax2.plot(
            param_values,
            info_gains,
            marker="o",
            color="tab:red",
            alpha=0.6,
            label="ΔH",
        )

    # If penalize mode, also show effective info gain
    if mode == "penalize":
        eff_gains = [
            pv.get("effective_info_gain", ig) for pv, ig in zip(per_value, info_gains)
        ]
        ax2.plot(
            param_values,
            eff_gains,
            marker="^",
            color="tab:green",
            alpha=0.8,
            linestyle="--",
            label="Effective ΔH",
        )

    ax2.set_ylabel("Information Gain (bits)", fontsize=13, color="tab:red")
    ax2.legend(loc="upper right", frameon=False, fontsize=9)

    ax1.set_title("Model Discrimination", fontsize=13)

    path = str(output_dir / "model_discrimination.png")
    fig.savefig(path)
    plt.close(fig)
    generated.append(path)

    return generated
