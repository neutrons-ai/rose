"""
Command-line interface for ROSE.

Provides subcommands for experiment design optimization:

    rose optimize   — Run Bayesian optimzation over parameter values
    rose inspect    — Display model parameters and their bounds
    rose report     — Generate plots from a results JSON file

Example usage::

    rose inspect examples/models/layer_a_on_b.yaml
    rose optimize examples/models/layer_a_on_b.yaml
    rose optimize examples/models/layer_a_on_b.yaml --output-dir my_results
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import click
import numpy as np

from rose import __version__

# Load .env before anything reads os.environ
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


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
@click.argument("model_file", type=click.Path(exists=True))
@click.option(
    "--data-file",
    type=click.Path(exists=True),
    default=None,
    help="Measurement data file (4 columns: Q, R, dR, dQ). "
    "Overrides the Q-grid settings in the model file.",
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
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=None,
    help="Max parallel worker processes (default: min(values, CPUs, 8))",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def optimize(
    model_file: str,
    data_file: str | None,
    output_dir: str,
    parallel: bool,
    workers: int | None,
    verbose: bool,
) -> None:
    """Optimise experiment design by maximising information gain.

    MODEL_FILE is a YAML or JSON file describing the layer stack,
    instrument settings (experiment section), and what to optimise
    (optimization section).

    All optimisation parameters (param, param_values, mcmc_steps, etc.)
    are read from the model file.  Use --data-file to override the
    Q-grid with a real measurement.
    """
    _setup_logging(verbose)

    from rose.planner import instrument as inst
    from rose.planner import optimizer
    from rose.planner.experiment_design import ExperimentDesigner
    from rose.planner.model_loader import (
        EXPERIMENT_DEFAULTS,
        OPTIMIZATION_DEFAULTS,
        load_experiment,
        load_model_description,
    )
    from rose.planner.report import make_report

    _validate_output_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load model description and extract sections
    desc = load_model_description(model_file)
    expt_cfg = {**EXPERIMENT_DEFAULTS, **desc.get("experiment", {})}
    opt_cfg = {**OPTIMIZATION_DEFAULTS, **desc.get("optimization", {})}

    # Validate required optimization keys
    if "param" not in opt_cfg:
        raise click.UsageError(
            "Model file must contain optimization.param "
            "(the parameter to optimise over)"
        )
    if "param_values" not in opt_cfg:
        raise click.UsageError(
            "Model file must contain optimization.param_values "
            "(list of candidate values to test)"
        )

    param = opt_cfg["param"]
    param_vals = [float(v) for v in opt_cfg["param_values"]]
    poi = opt_cfg.get("parameters_of_interest")
    if isinstance(poi, list):
        poi = [str(s) for s in poi]
    num_realizations = int(opt_cfg["num_realizations"])
    mcmc_steps = int(opt_cfg["mcmc_steps"])
    entropy_method = str(opt_cfg["entropy_method"])

    click.echo("Starting experiment design optimisation...")
    click.echo(f"  Model:       {model_file}")
    click.echo(f"  Parameter:   {param}")
    click.echo(f"  Values:      {param_vals}")
    click.echo(f"  Realizations: {num_realizations}")
    click.echo(f"  MCMC steps:  {mcmc_steps}")
    click.echo(f"  Method:      {entropy_method}")
    click.echo(f"  Mode:        {'parallel' if parallel else 'sequential'}")

    # CLI --data-file overrides YAML experiment.data_file
    effective_data_file = data_file or expt_cfg.get("data_file")

    # Build simulator
    if effective_data_file:
        simulator = inst.InstrumentSimulator(data_file=effective_data_file)
    else:
        q_min = float(expt_cfg["q_min"])
        q_max = float(expt_cfg["q_max"])
        q_points = int(expt_cfg["q_points"])
        dq_over_q = float(expt_cfg["dq_over_q"])
        relative_error = float(expt_cfg["relative_error"])
        q = np.logspace(np.log10(q_min), np.log10(q_max), q_points)
        dq = dq_over_q * q
        simulator = inst.InstrumentSimulator(
            q_values=q, dq_values=dq, relative_error=relative_error
        )
        click.echo(f"  Q range:     {q_min}–{q_max} Å⁻¹ ({q_points} points)")
        click.echo(f"  dQ/Q:        {dq_over_q}")
        click.echo(f"  dR/R:        {relative_error}")

    experiment = load_experiment(model_file, simulator.q_values, simulator.dq_values)
    designer = ExperimentDesigner(
        experiment, simulator=simulator, parameters_of_interest=poi
    )

    h_prior = designer.prior_entropy()
    click.echo(f"\n  Prior entropy: {h_prior:.4f} bits")

    if verbose:
        click.echo(str(designer))

    click.echo("\nRunning optimisation...\n")

    if parallel:
        results, simulated_data = optimizer.optimize_parallel(
            designer,
            param_to_optimize=param,
            param_values=param_vals,
            realizations=num_realizations,
            mcmc_steps=mcmc_steps,
            entropy_method=entropy_method,
            max_workers=workers,
        )
    else:
        results, simulated_data = optimizer.optimize(
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

    return result_dict  # for programmatic use


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


# ── check-llm ────────────────────────────────────────────────────

# ALCF inference endpoint base URLs per cluster
_ALCF_CLUSTER_URLS: dict[str, str] = {
    "sophia": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    "metis": "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1",
}

_ALCF_AUTH_SCRIPT_URL = (
    "https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/"
    "refs/heads/main/inference_auth_token.py"
)


def _get_alcf_token() -> str:
    """Obtain a Globus access token for ALCF inference endpoints.

    Resolution order:
      1. ``ALCF_ACCESS_TOKEN`` environment variable.
      2. ``globus_sdk`` — reuses cached / refresh tokens.
      3. Run ``inference_auth_token.py get_access_token`` (subprocess).

    Raises:
        RuntimeError: If no token can be obtained.
    """
    import subprocess
    import sys

    # 1. Explicit env-var
    token = os.environ.get("ALCF_ACCESS_TOKEN")
    if token:
        return token

    # 2. globus_sdk
    try:
        import globus_sdk

        app_name = os.environ.get("GLOBUS_APP_NAME", "inference_app")
        client_id = os.environ.get(
            "GLOBUS_AUTH_CLIENT_ID",
            "REDACTED_GLOBUS_CLIENT_ID",
        )
        gateway_id = os.environ.get(
            "GLOBUS_GATEWAY_CLIENT_ID",
            "REDACTED_GLOBUS_GATEWAY_ID",
        )
        scope = os.environ.get(
            "GLOBUS_GATEWAY_SCOPE",
            f"https://auth.globus.org/scopes/{gateway_id}/action_all",
        )
        app = globus_sdk.UserApp(
            app_name,
            client_id=client_id,
            scope_requirements={gateway_id: [scope]},
            config=globus_sdk.GlobusAppConfig(
                request_refresh_tokens=True,
            ),
        )
        auth = app.get_authorizer(gateway_id)
        auth.ensure_valid_token()
        return auth.access_token
    except ImportError:
        pass
    except Exception:
        pass

    # 3. subprocess fallback
    try:
        result = subprocess.run(
            [sys.executable, "inference_auth_token.py", "get_access_token"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    raise RuntimeError(
        "Could not obtain ALCF access token. Options:\n"
        "  1. Set ALCF_ACCESS_TOKEN in environment or .env\n"
        "  2. Install globus_sdk and authenticate\n"
        "  3. Download the auth script:\n"
        f"     wget {_ALCF_AUTH_SCRIPT_URL}\n"
        "     python inference_auth_token.py authenticate"
    )


def _alcf_token_available() -> bool:
    """Return ``True`` if an ALCF token can be obtained silently."""
    try:
        _get_alcf_token()
        return True
    except Exception:
        return False


def _get_llm_config() -> dict:
    """Read LLM configuration from environment variables.

    Supports providers: ``openai``, ``alcf``, ``local``.
    When ``LLM_PROVIDER=alcf``, the base URL defaults to the ALCF
    Sophia endpoint and authentication uses Globus tokens (not API keys).

    Returns:
        Dict with keys: provider, model, api_key, base_url,
        temperature, max_tokens, alcf_cluster.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai")
    model = os.environ.get("LLM_MODEL", "gpt-4o")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base_url = os.environ.get("LLM_BASE_URL", "")
    temperature = os.environ.get("LLM_TEMPERATURE", "0.2")
    max_tokens = os.environ.get("LLM_MAX_TOKENS", "4096")
    alcf_cluster = ""

    # ALCF-specific defaults
    if provider == "alcf":
        alcf_cluster = os.environ.get("ALCF_CLUSTER", "sophia").lower()
        if not base_url:
            base_url = _ALCF_CLUSTER_URLS.get(
                alcf_cluster,
                _ALCF_CLUSTER_URLS["sophia"],
            )

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "alcf_cluster": alcf_cluster,
    }


def _mask_key(key: str) -> str:
    """Return a masked version of an API key for display."""
    if not key:
        return ""
    if len(key) <= 8:
        return "••••"
    return "••••" + key[-4:]


@main.command("check-llm")
@click.option("--no-test", is_flag=True, help="Skip the live connection test")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--fix",
    is_flag=True,
    help="Attempt to fix issues (e.g. download ALCF auth script)",
)
def check_llm(no_test: bool, output_json: bool, fix: bool) -> None:
    """Check LLM configuration and connectivity.

    Shows which provider and model are configured, whether credentials
    are present, and (unless --no-test) sends a tiny test prompt to
    verify the connection works end-to-end.

    Use --fix to automatically download and run the ALCF authentication
    helper when the provider is 'alcf' and credentials are missing or
    expired.

    \b
    Environment variables used:
        LLM_PROVIDER     openai | alcf | local  (default: openai)
        LLM_MODEL        model name             (default: gpt-4o)
        LLM_API_KEY      API key (or OPENAI_API_KEY)
        LLM_BASE_URL     base URL (for local / custom endpoints)
        LLM_TEMPERATURE  sampling temperature   (default: 0.2)

    \b
    Examples:
        rose check-llm
        rose check-llm --no-test
        rose check-llm --json
        rose check-llm --fix
    """
    import sys

    config = _get_llm_config()
    is_alcf = config["provider"] == "alcf"
    has_key = bool(config["api_key"])
    has_token = _alcf_token_available() if is_alcf else False
    creds_ok = has_token if is_alcf else (has_key or config["provider"] == "local")

    # Check that langchain-openai is installed
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
    except ImportError:
        deps_ok = False
        deps_msg = "langchain-openai not installed (pip install rose[llm])"
    else:
        deps_ok = True
        deps_msg = ""

    # --- JSON output ---
    if output_json:
        ok = False
        message = ""

        if not deps_ok:
            ok, message = False, deps_msg
        elif not creds_ok:
            ok, message = (
                False,
                ("ALCF token not available" if is_alcf else "API key not set"),
            )
        elif no_test:
            ok, message = True, "Credentials present (live test skipped)"
        else:
            ok, message = _test_llm_connection(config)

        result = {
            "provider": config["provider"],
            "model": config["model"],
            "has_api_key": has_key,
            "has_token": has_token,
            "base_url": config["base_url"] or None,
            "temperature": config["temperature"],
            "deps_installed": deps_ok,
            "ok": ok,
            "message": message,
        }
        if is_alcf:
            result["alcf_cluster"] = config["alcf_cluster"]
        click.echo(json.dumps(result, indent=2))
        sys.exit(0 if ok else 1)

    # --- Human-readable output ---
    click.echo()
    click.echo(click.style("  LLM Configuration Check", fg="blue", bold=True))
    click.echo(click.style("  " + "─" * 40, fg="blue"))
    click.echo()
    click.echo(f"    Provider:    {config['provider']}")
    click.echo(f"    Model:       {config['model']}")

    if is_alcf:
        token_display = (
            click.style("✓ available", fg="green")
            if has_token
            else click.style("NOT AVAILABLE", fg="red")
        )
        click.echo(f"    Token:       {token_display}")
        click.echo(f"    Cluster:     {config['alcf_cluster']}")
    else:
        if has_key:
            click.echo(f"    API key:     {_mask_key(config['api_key'])}")
        else:
            click.echo(f"    API key:     {click.style('NOT SET', fg='red')}")

    if config["base_url"]:
        click.echo(f"    Base URL:    {config['base_url']}")
    click.echo(f"    Temperature: {config['temperature']}")
    click.echo()

    if not deps_ok:
        click.echo(click.style(f"  ✗ {deps_msg}", fg="red", bold=True))
        click.echo()
        sys.exit(1)

    if not creds_ok:
        if is_alcf:
            click.echo(click.style("  ✗ ALCF token not available", fg="red", bold=True))
            click.echo()
            _show_alcf_auth_hint(offer_fix=fix)
        else:
            click.echo(click.style("  ✗ API key not set", fg="red", bold=True))
            click.echo()
            click.echo("    Set an API key:")
            click.echo("      export LLM_API_KEY=<your-key>")
            click.echo("    or add to .env:")
            click.echo(f"      LLM_PROVIDER={config['provider']}")
            click.echo("      LLM_API_KEY=<your-key>")
        click.echo()
        sys.exit(1)

    if no_test:
        click.echo(
            click.style("  ✓ Credentials present (skipped live test)", fg="green")
        )
        click.echo()
        return

    click.echo("    Testing connection...", nl=False)
    ok, msg = _test_llm_connection(config)
    if ok:
        click.echo(click.style(" ✓ Connected", fg="green"))
    else:
        click.echo(click.style(f" ✗ {msg}", fg="red"))
        if (
            is_alcf
            and "token" in msg.lower()
            or "401" in msg
            or "unauthorized" in msg.lower()
        ):
            click.echo()
            click.echo(
                click.style(
                    "    ALCF tokens expire periodically. Re-authenticate:",
                    fg="yellow",
                )
            )
            click.echo()
            _show_alcf_auth_hint(offer_fix=fix)
    click.echo()
    sys.exit(0 if ok else 1)


def _show_alcf_auth_hint(*, offer_fix: bool = False) -> None:
    """Print ALCF authentication instructions."""
    if offer_fix and click.confirm(
        "    Download and run the ALCF auth script now?", default=True
    ):
        _alcf_authenticate()
        return

    click.echo("      # Download the authentication helper script")
    click.echo(f"      wget {_ALCF_AUTH_SCRIPT_URL}")
    click.echo()
    click.echo("      # Authenticate with your Globus account")
    click.echo("      python inference_auth_token.py authenticate")
    click.echo()
    click.echo("      # Then set the token:")
    click.echo(
        "      export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)"
    )


def _alcf_authenticate() -> bool:
    """Download the ALCF auth helper and run ``authenticate``."""
    import subprocess
    import sys
    import tempfile
    import urllib.request

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "inference_auth_token.py"
        click.echo()
        click.echo("    Downloading inference_auth_token.py …", nl=False)
        try:
            urllib.request.urlretrieve(_ALCF_AUTH_SCRIPT_URL, script)  # noqa: S310
            click.echo(click.style(" done", fg="green"))
        except Exception as exc:
            click.echo(click.style(f" failed: {exc}", fg="red"))
            return False

        click.echo("    Launching Globus authentication (a browser window may open)…")
        click.echo()
        result = subprocess.run([sys.executable, str(script), "authenticate"])
        if result.returncode != 0:
            click.echo()
            click.echo(
                click.style("    Authentication script exited with an error.", fg="red")
            )
            return False

        click.echo()
        click.echo(click.style("    ✓ Authentication complete.", fg="green"))
        click.echo("    You can now obtain a token by running:")
        click.echo(f"      python {script.name} get_access_token")
        click.echo("    or set ALCF_ACCESS_TOKEN in your environment.")
        return True


def _test_llm_connection(config: dict) -> tuple[bool, str]:
    """Send a tiny test prompt to verify LLM connectivity.

    Returns:
        ``(ok, message)`` tuple.
    """
    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "model": config["model"],
        "temperature": float(config["temperature"]),
        "max_tokens": 256,
    }

    # Resolve credentials
    if config["provider"] == "alcf":
        try:
            kwargs["api_key"] = _get_alcf_token()
        except RuntimeError as e:
            return False, str(e)
    elif config["api_key"]:
        kwargs["api_key"] = config["api_key"]

    if config["base_url"]:
        kwargs["base_url"] = config["base_url"]

    try:
        llm = ChatOpenAI(**kwargs)
        response = llm.invoke("Reply with only the word 'OK'")
        content = getattr(response, "content", None) or ""
        if not content.strip():
            return False, "LLM returned an empty response"
        return True, "LLM connected successfully"
    except Exception as e:
        error_msg = str(e)
        error_lower = error_msg.lower()

        if (
            any(w in error_lower for w in ("quota", "rate", "limit"))
            or "429" in error_msg
        ):
            return False, "API quota/rate limit exceeded"
        if (
            "401" in error_msg
            or "unauthorized" in error_lower
            or "api key" in error_lower
            and "invalid" in error_lower
        ):
            return False, "Invalid API key"
        if "not found" in error_lower or "404" in error_msg:
            return False, f"Model '{config['model']}' not found"
        if "connection" in error_lower or "connect" in error_lower:
            short = "Connection failed"
            if config["base_url"]:
                short += f" ({config['base_url']})"
            return False, short

        # Generic — truncate long messages
        if len(error_msg) > 120:
            error_msg = error_msg[:120] + "..."
        return False, f"Error: {error_msg}"


# ── plan ─────────────────────────────────────────────────────────


@main.command()
@click.argument("query_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for the generated YAML model (default: stdout)",
)
@click.option(
    "--model-name",
    default=None,
    help="LLM model identifier (default: from .env or gpt-4o)",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="LLM sampling temperature (default: from .env or 0.2)",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def plan(
    query_file: str,
    output: str | None,
    model_name: str | None,
    temperature: float | None,
    verbose: bool,
) -> None:
    """Generate a ROSE YAML model from a plain-text description using an LLM.

    QUERY_FILE is a text file (.txt) or YAML file (.yaml) containing
    a plain-language description of the sample, hypothesis, and what
    to optimise.  The LLM translates it into a full ROSE model file.

    Requires the ``llm`` extras: ``pip install rose[llm]``
    """
    _setup_logging(verbose)

    from rose.modeler.llm_generator import generate_model_yaml
    from rose.modeler.schema import load_query

    config = _get_llm_config()
    model_name = model_name or config["model"]
    temperature = (
        temperature if temperature is not None else float(config["temperature"])
    )

    query = load_query(query_file)
    desc_preview = query.description.strip()[:80]
    click.echo(f"Loaded query: {query_file}")
    click.echo(f"  Description: {desc_preview}...")
    click.echo(f"  LLM model: {model_name}")

    api_key = config["api_key"] or None
    if config["provider"] == "alcf" and not api_key:
        try:
            api_key = _get_alcf_token()
        except RuntimeError as e:
            raise click.ClickException(f"ALCF authentication failed: {e}")  # noqa: B904

    yaml_text = generate_model_yaml(
        query.description,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=config["base_url"] or None,
        max_tokens=int(config["max_tokens"]),
    )

    if output:
        out_path = _validate_output_path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_text)
        click.echo(f"\nModel saved to: {out_path}")
    else:
        click.echo("\n--- Generated ROSE YAML model ---")
        click.echo(yaml_text)
        click.echo("--- End of model ---")


# ── plan-and-optimize ────────────────────────────────────────────


@main.command("plan-and-optimize")
@click.argument("query_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results",
    help="Directory for output files",
)
@click.option(
    "--model-name",
    default=None,
    help="LLM model identifier (default: from .env or gpt-4o)",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="LLM sampling temperature (default: from .env or 0.2)",
)
@click.option(
    "--data-file",
    type=click.Path(exists=True),
    default=None,
    help="Optional measured data file (4-column Q, dQ, R, dR)",
)
@click.option(
    "--parallel/--sequential",
    default=True,
    help="Run in parallel or sequential mode",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=None,
    help="Max parallel worker processes",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def plan_and_optimize(
    query_file: str,
    output_dir: str,
    model_name: str | None,
    temperature: float | None,
    data_file: str | None,
    parallel: bool,
    workers: int | None,
    verbose: bool,
) -> None:
    """Generate a model from a plain-text description then run optimisation.

    This combines ``rose plan`` and ``rose optimize`` into a single
    command.  The LLM generates a YAML model from the description,
    saves it to the output directory, then runs the optimiser on it.

    QUERY_FILE is a text file (.txt) or YAML file (.yaml) with a
    plain-language sample description.
    Requires ``pip install rose[llm]``.
    """
    _setup_logging(verbose)

    from rose.modeler.llm_generator import generate_model_yaml
    from rose.modeler.schema import load_query

    config = _get_llm_config()
    model_name = model_name or config["model"]
    temperature = (
        temperature if temperature is not None else float(config["temperature"])
    )

    out_path = _validate_output_path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate model
    query = load_query(query_file)
    click.echo("Step 1/2: Generating YAML model via LLM...")
    click.echo(f"  Query: {query_file}")
    desc_preview = query.description.strip()[:80]
    click.echo(f"  Description: {desc_preview}...")

    api_key = config["api_key"] or None
    if config["provider"] == "alcf" and not api_key:
        try:
            api_key = _get_alcf_token()
        except RuntimeError as e:
            raise click.ClickException(f"ALCF authentication failed: {e}")  # noqa: B904

    yaml_text = generate_model_yaml(
        query.description,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=config["base_url"] or None,
        max_tokens=int(config["max_tokens"]),
    )

    model_path = out_path / "generated_model.yaml"
    model_path.write_text(yaml_text)
    click.echo(f"  Saved model: {model_path}")

    # Step 2: Run optimize using the generated model
    click.echo("\nStep 2/2: Running optimisation...")
    ctx = click.get_current_context()
    ctx.invoke(
        optimize,
        model_file=str(model_path),
        data_file=data_file,
        output_dir=output_dir,
        parallel=parallel,
        workers=workers,
        verbose=verbose,
    )


# ── serve ────────────────────────────────────────────────────────


@main.command()
@click.argument(
    "results_dir",
    type=click.Path(exists=True),
    default="results",
)
@click.option(
    "--port",
    "-p",
    default=5000,
    type=int,
    help="Port to run the web server on (default: 5000)",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open a browser automatically",
)
def serve(results_dir: str, port: int, no_browser: bool) -> None:
    """Launch the ROSE web interface to browse optimization results.

    RESULTS_DIR is the directory containing optimization output
    (with ``optimization_results.json`` files).  Defaults to
    ``results/`` in the current directory.

    \b
    Examples:
        rose serve
        rose serve results/
        rose serve my_output --port 8080
        rose serve results/ --no-browser
    """
    from rose.web import create_app

    click.echo(click.style("═" * 60, fg="blue"))
    click.echo(click.style("  ROSE – Results Viewer", fg="blue", bold=True))
    click.echo(click.style("═" * 60, fg="blue"))
    click.echo()
    click.echo(f"  Results dir: {results_dir}")
    click.echo(f"  URL:         http://127.0.0.1:{port}")
    click.echo()

    app = create_app(results_dir)

    if not no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=[f"http://127.0.0.1:{port}"]).start()

    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
