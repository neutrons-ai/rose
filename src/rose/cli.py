"""
Command-line interface for ROSE.

Provides subcommands for experiment design optimization:

    rose optimize   — Run Bayesian optimzation over parameter values
    rose inspect    — Display model parameters and their bounds
    rose report     — Generate plots from a results JSON file

Example usage::

    rose inspect examples/models/layer_a_on_b.yaml
    rose optimize --model-file examples/models/layer_a_on_b.yaml \\
        --param "layer_B thickness" --param-values "20,40,60,80"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import click
import numpy as np

from rose import __version__


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _validate_output_path(path_str: str) -> Path:
    """Validate that an output path does not contain path traversal.

    Rejects paths with ``..`` components to prevent writing outside
    the intended directory tree.  This is especially important when
    the CLI is invoked programmatically from a web backend (Phase 2+).

    Raises:
        click.BadParameter: On path traversal attempt.
    """
    p = Path(path_str)
    if ".." in p.parts:
        raise click.BadParameter(
            f"Path traversal ('..') not allowed in output path: {path_str}"
        )
    return p.resolve()


# ── top-level group ──────────────────────────────────────────────


@click.group()
@click.version_option(version=__version__)
def main():
    """ROSE — Reflectivity Open Science Engine.

    Tools for Bayesian experiment design optimisation
    of neutron reflectometry measurements.
    """


# ── inspect ──────────────────────────────────────────────────────


@main.command()
@click.argument("model_file", type=click.Path(exists=True))
@click.option("--verbose", is_flag=True, help="Show all parameters including fixed")
def inspect(model_file: str, verbose: bool) -> None:
    """Display model parameters and their bounds.

    MODEL_FILE is a YAML or JSON file describing the layer stack.
    """
    from rose.planner.model_loader import inspect_model

    info = inspect_model(model_file)

    click.echo("Variable (fitted) parameters:")
    click.echo(f"  {'Name':<30} {'Value':<10} {'Bounds'}")
    click.echo("  " + "-" * 60)
    for p in info["variable"]:
        click.echo(f"  {p['name']:<30} {p['value']:<10.4g} {p['bounds']}")

    if verbose and info["fixed"]:
        click.echo("\nFixed parameters:")
        click.echo(f"  {'Name':<30} {'Value':<10}")
        click.echo("  " + "-" * 40)
        for p in info["fixed"]:
            click.echo(f"  {p['name']:<30} {p['value']:<10.4g}")


# ── optimize ─────────────────────────────────────────────────────


@main.command()
@click.option(
    "--model-file",
    type=click.Path(exists=True),
    required=True,
    help="YAML or JSON model file describing the layer stack",
)
@click.option(
    "--data-file",
    type=click.Path(exists=True),
    default=None,
    help="Measurement data file (4 columns: Q, R, dR, dQ). "
    "If omitted, a default Q grid is used.",
)
@click.option("--param", type=str, required=True, help="Parameter to optimise over")
@click.option(
    "--param-values",
    type=str,
    required=True,
    help="Comma-separated values to test (e.g. '20,40,60')",
)
@click.option(
    "--parameters-of-interest",
    type=str,
    default=None,
    help="Comma-separated names for marginal entropy (default: all variable params)",
)
@click.option(
    "--num-realizations",
    type=click.IntRange(1, 100),
    default=3,
    help="Noise realizations (1–100)",
)
@click.option(
    "--mcmc-steps",
    type=click.IntRange(100, 100_000),
    default=2000,
    help="MCMC steps after burn-in (100–100,000)",
)
@click.option(
    "--entropy-method",
    type=click.Choice(["mvn", "kdn"]),
    default="kdn",
    help="Entropy calculation method",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results",
    help="Directory for output files",
)
@click.option(
    "--parallel/--sequential",
    default=True,
    help="Run in parallel or sequential mode",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def optimize(
    model_file: str,
    data_file: str | None,
    param: str,
    param_values: str,
    parameters_of_interest: str | None,
    num_realizations: int,
    mcmc_steps: int,
    entropy_method: str,
    output_dir: str,
    parallel: bool,
    verbose: bool,
) -> None:
    """Optimise experiment design by maximising information gain.

    Evaluates a grid of parameter values and reports which value
    yields the greatest expected information gain about the
    parameters of interest.
    """
    _setup_logging(verbose)

    from rose.planner import instrument as inst
    from rose.planner import optimizer
    from rose.planner.experiment_design import ExperimentDesigner
    from rose.planner.model_loader import load_experiment
    from rose.planner.report import make_report

    _validate_output_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    param_vals = [float(x.strip()) for x in param_values.split(",")]
    poi = (
        [s.strip() for s in parameters_of_interest.split(",")]
        if parameters_of_interest
        else None
    )

    click.echo("Starting experiment design optimisation...")
    click.echo(f"  Model:       {model_file}")
    click.echo(f"  Parameter:   {param}")
    click.echo(f"  Values:      {param_vals}")
    click.echo(f"  Realizations: {num_realizations}")
    click.echo(f"  MCMC steps:  {mcmc_steps}")
    click.echo(f"  Method:      {entropy_method}")
    click.echo(f"  Mode:        {'parallel' if parallel else 'sequential'}")

    # Build simulator
    if data_file:
        simulator = inst.InstrumentSimulator(data_file=data_file)
    else:
        simulator = inst.InstrumentSimulator()

    experiment = load_experiment(model_file, simulator.q_values, simulator.dq_values)
    designer = ExperimentDesigner(
        experiment, simulator=simulator, parameters_of_interest=poi
    )

    h_prior = designer.prior_entropy()
    click.echo(f"\n  Prior entropy: {h_prior:.4f} bits")

    if verbose:
        click.echo(str(designer))

    click.echo("\nRunning optimisation...\n")

    run_fn = optimizer.optimize_parallel if parallel else optimizer.optimize
    results, simulated_data = run_fn(
        designer,
        param_to_optimize=param,
        param_values=param_vals,
        realizations=num_realizations,
        mcmc_steps=mcmc_steps,
        entropy_method=entropy_method,
    )

    # Display results
    click.echo(f"\n{'=' * 55}")
    click.echo("OPTIMISATION RESULTS")
    click.echo(f"{'=' * 55}")
    click.echo(f"{'Value':>10}   {'ΔH (bits)':>12}   {'± std':>10}")
    click.echo("-" * 55)
    for r in results:
        click.echo(f"{r[0]:>10.3f}   {r[1]:>12.4f}   {r[2]:>10.4f}")

    best_idx = int(np.argmax([r[1] for r in results]))
    best_val, best_gain, best_std = results[best_idx]
    click.echo(f"\nOptimal value: {best_val:.3f}")
    click.echo(f"Max ΔH:        {best_gain:.4f} ± {best_std:.4f} bits")

    # Save JSON
    result_dict = {
        "parameter": param,
        "parameter_values": param_vals,
        "results": results,
        "simulated_data": simulated_data,
        "optimal_value": best_val,
        "max_information_gain": best_gain,
        "max_information_gain_std": best_std,
        "prior_entropy": h_prior,
        "settings": {
            "num_realizations": num_realizations,
            "mcmc_steps": mcmc_steps,
            "entropy_method": entropy_method,
            "parallel": parallel,
        },
    }
    json_path = os.path.join(output_dir, "optimization_results.json")
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    click.echo(f"\nResults saved to: {json_path}")

    # Generate plots
    make_report(json_file=json_path, output_dir=output_dir)
    click.echo(f"Plots saved to: {output_dir}/")

    # ASCII graph
    _print_ascii_graph(results)


def _print_ascii_graph(results: list[list[float]]) -> None:
    """Print a simple ASCII bar chart of information gain."""
    gains = [r[1] for r in results]
    max_gain = max(gains) if gains else 1.0
    scale = 40 / max_gain if max_gain > 0 else 1.0

    click.echo(f"\n{'Value':<8} | Information Gain")
    click.echo("-" * 60)
    for r in results:
        bar = "#" * int(r[1] * scale)
        click.echo(f"{r[0]:>6.2f}   | {bar} ({r[1]:.3f} ± {r[2]:.3f})")


# ── report ───────────────────────────────────────────────────────


@main.command()
@click.option(
    "--result-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to optimization_results.json",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Directory to save generated plots",
)
def report(result_file: str, output_dir: str) -> None:
    """Generate plots from a previous optimisation run."""
    from rose.planner.report import make_report

    _validate_output_path(output_dir)
    paths = make_report(json_file=result_file, output_dir=output_dir)
    click.echo(f"Generated {len(paths)} plots in {output_dir}/")


if __name__ == "__main__":
    main()
