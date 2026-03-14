"""Flask blueprint – page routes and JSON API endpoints for ROSE."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
)

from .data import ResultData, list_results

logger = logging.getLogger(__name__)

bp = Blueprint(
    "web",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/web",
)


def _results_dir() -> str | None:
    return current_app.config.get("RESULTS_DIR")


# ── Page routes ──────────────────────────────────────────────────


@bp.route("/")
def index():
    """Landing page – browse result directories."""
    results_dir = _results_dir()
    items = []
    if results_dir:
        for rd in list_results(results_dir):
            summary = rd.get_summary()
            summary["has_model"] = rd.get_model_yaml() is not None
            items.append(summary)

    return render_template(
        "index.html",
        results=items,
        results_dir=results_dir or "(not set)",
        active_tab="results",
    )


@bp.route("/results/<result_id>")
def result_detail(result_id: str):
    """Detail page for a single optimization result."""
    results_dir = _results_dir()
    if not results_dir:
        return "Results directory not configured", 404

    rd = _find_result(results_dir, result_id)
    if rd is None:
        return f"Result '{result_id}' not found", 404

    summary = rd.get_summary()
    info_gain = rd.get_info_gain()

    return render_template(
        "result.html",
        result_id=result_id,
        summary=summary,
        param_values=info_gain["values"],
        settings=rd.get_settings(),
        has_model=rd.get_model_yaml() is not None,
        active_tab="results",
    )


@bp.route("/results/<result_id>/model")
def model_view(result_id: str):
    """Model viewer page."""
    results_dir = _results_dir()
    if not results_dir:
        return "Results directory not configured", 404

    rd = _find_result(results_dir, result_id)
    if rd is None:
        return f"Result '{result_id}' not found", 404

    return render_template(
        "model.html",
        result_id=result_id,
        model_yaml=rd.get_model_yaml(),
        settings=rd.get_settings(),
        active_tab="results",
    )


# ── JSON API ─────────────────────────────────────────────────────


@bp.route("/api/results")
def api_results():
    """List all result summaries as JSON."""
    results_dir = _results_dir()
    if not results_dir:
        return jsonify([])

    items = []
    for rd in list_results(results_dir):
        summary = rd.get_summary()
        summary["has_model"] = rd.get_model_yaml() is not None
        items.append(summary)
    return jsonify(items)


@bp.route("/api/results/<result_id>/info-gain")
def api_info_gain(result_id: str):
    """Information gain curve data."""
    rd = _get_result_or_404(result_id)
    if rd is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(rd.get_info_gain())


@bp.route("/api/results/<result_id>/reflectivity")
def api_reflectivity(result_id: str):
    """Reflectivity data for a parameter value index."""
    rd = _get_result_or_404(result_id)
    if rd is None:
        return jsonify({"error": "not found"}), 404

    idx = request.args.get("index", 0, type=int)
    return jsonify(rd.get_reflectivity(idx))


@bp.route("/api/results/<result_id>/sld")
def api_sld(result_id: str):
    """SLD profile data for a parameter value index."""
    rd = _get_result_or_404(result_id)
    if rd is None:
        return jsonify({"error": "not found"}), 404

    idx = request.args.get("index", 0, type=int)
    return jsonify(rd.get_sld(idx))


@bp.route("/api/results/<result_id>/settings")
def api_settings(result_id: str):
    """Optimization settings."""
    rd = _get_result_or_404(result_id)
    if rd is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(rd.get_settings())


@bp.route("/api/results/<result_id>/summary")
def api_summary(result_id: str):
    """Result summary."""
    rd = _get_result_or_404(result_id)
    if rd is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(rd.get_summary())


# ── Interactive page routes ──────────────────────────────────────


@bp.route("/optimize")
def optimize_page():
    """Optimization setup form."""
    return render_template("optimize.html", active_tab="optimize")


@bp.route("/plan")
def plan_page():
    """Planning (LLM text-to-model) setup form."""
    return render_template("plan.html", active_tab="plan")


# ── File browsing API ────────────────────────────────────────────


def _safe_browse_path(raw: str) -> Path | None:
    """Resolve and validate a browse path.  Returns None if unsafe."""
    try:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            return None
        return p
    except Exception:
        return None


@bp.route("/api/browse-files")
def api_browse_files():
    """List files and directories at a given path.

    Query params:
        path — directory to list (default: home dir)
        ext  — optional extension filter, e.g. ".yaml"
    """
    raw = request.args.get("path", str(Path.home()))
    ext = request.args.get("ext", "")
    target = _safe_browse_path(raw)
    if target is None:
        return jsonify({"error": "Path does not exist"}), 400
    if not target.is_dir():
        target = target.parent

    entries: list[dict] = []
    try:
        for child in sorted(
            target.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        ):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                entries.append({"name": child.name, "is_dir": True, "path": str(child)})
            elif not ext or child.suffix.lower() == ext.lower():
                entries.append(
                    {"name": child.name, "is_dir": False, "path": str(child)}
                )
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    parent = str(target.parent) if target.parent != target else None
    return jsonify({"current": str(target), "parent": parent, "entries": entries})


@bp.route("/api/browse-dirs")
def api_browse_dirs():
    """List only directories at a given path.

    Query params:
        path — directory to list (default: cwd)
    """
    raw = request.args.get("path", str(Path.cwd()))
    target = _safe_browse_path(raw)
    if target is None:
        return jsonify({"error": "Path does not exist"}), 400
    if not target.is_dir():
        target = target.parent

    entries: list[dict] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                entries.append({"name": child.name, "path": str(child)})
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    parent = str(target.parent) if target.parent != target else None
    return jsonify({"current": str(target), "parent": parent, "entries": entries})


# ── Background job API ───────────────────────────────────────────


def _get_jobs() -> tuple[dict, threading.Lock]:
    return current_app.config["JOBS"], current_app.config["JOBS_LOCK"]


@bp.route("/api/jobs/optimize", methods=["POST"])
def api_start_optimize():
    """Start a background optimization job.

    Expects JSON body::

        {
            "model_file": "/abs/path/to/model.yaml",
            "output_dir": "/abs/path/to/output",
            "parallel": true,
            "workers": null,
            "data_file": null
        }
    """
    body = request.get_json(silent=True) or {}
    model_file = (body.get("model_file") or "").strip()
    output_dir = (body.get("output_dir") or "").strip()

    errors = []
    if not model_file or not Path(model_file).is_file():
        errors.append("model_file: file does not exist")
    if not output_dir:
        errors.append("output_dir is required")
    if errors:
        return jsonify({"errors": errors}), 400

    parallel = body.get("parallel", True)
    workers = body.get("workers")
    data_file = (body.get("data_file") or "").strip() or None
    if data_file and not Path(data_file).is_file():
        return jsonify({"errors": ["data_file: file does not exist"]}), 400

    job_id = str(uuid.uuid4())[:8]
    jobs, lock = _get_jobs()

    with lock:
        jobs[job_id] = {
            "id": job_id,
            "type": "optimize",
            "status": "running",
            "progress": "Starting optimization...",
            "model_file": model_file,
            "output_dir": output_dir,
            "error": None,
        }

    app = current_app._get_current_object()

    def _run():
        try:
            _run_optimize_job(
                app, job_id, model_file, output_dir, parallel, workers, data_file
            )
        except Exception as exc:
            logger.exception("Optimize job %s failed", job_id)
            with lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "started"})


@bp.route("/api/jobs/plan", methods=["POST"])
def api_start_plan():
    """Start a background plan-and-optimize job.

    Expects JSON body::

        {
            "description": "Sample description text...",
            "output_dir": "/abs/path",
            "data_file": null,
            "parallel": true,
            "workers": null
        }
    """
    body = request.get_json(silent=True) or {}
    description = (body.get("description") or "").strip()
    output_dir = (body.get("output_dir") or "").strip()

    errors = []
    if not description or len(description) < 10:
        errors.append("description must be at least 10 characters")
    if not output_dir:
        errors.append("output_dir is required")
    if errors:
        return jsonify({"errors": errors}), 400

    parallel = body.get("parallel", True)
    workers = body.get("workers")
    data_file = (body.get("data_file") or "").strip() or None
    if data_file and not Path(data_file).is_file():
        return jsonify({"errors": ["data_file: file does not exist"]}), 400

    job_id = str(uuid.uuid4())[:8]
    jobs, lock = _get_jobs()

    with lock:
        jobs[job_id] = {
            "id": job_id,
            "type": "plan",
            "status": "running",
            "progress": "Generating model via LLM...",
            "output_dir": output_dir,
            "error": None,
        }

    app = current_app._get_current_object()

    def _run():
        try:
            _run_plan_job(
                app, job_id, description, output_dir, parallel, workers, data_file
            )
        except Exception as exc:
            logger.exception("Plan job %s failed", job_id)
            with lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "started"})


@bp.route("/api/jobs/<job_id>/status")
def api_job_status(job_id: str):
    """Return current status of a background job."""
    jobs, lock = _get_jobs()
    with lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        return jsonify({k: v for k, v in job.items() if not k.startswith("_")})


# ── Helpers ──────────────────────────────────────────────────────


def _find_result(results_dir: str, result_id: str) -> ResultData | None:
    """Find a result by ID, checking for path traversal."""
    # Reject path traversal
    if ".." in result_id or "/" in result_id or "\\" in result_id:
        return None

    # Check if the results_dir itself matches
    base = Path(results_dir)
    if base.name == result_id and (base / "optimization_results.json").exists():
        return ResultData(base)

    # Check subdirectory
    candidate = base / result_id
    if candidate.is_dir() and (candidate / "optimization_results.json").exists():
        return ResultData(candidate)

    return None


def _get_result_or_404(result_id: str) -> ResultData | None:
    """Resolve a result by ID, returning None if not found."""
    results_dir = _results_dir()
    if not results_dir:
        return None
    return _find_result(results_dir, result_id)


# ── Background job runners ───────────────────────────────────────


def _update_job(app, job_id: str, **kwargs) -> None:
    """Thread-safe update of a job's state dict."""
    with app.config["JOBS_LOCK"]:
        job = app.config["JOBS"].get(job_id)
        if job:
            job.update(kwargs)


def _run_optimize_job(
    app,
    job_id: str,
    model_file: str,
    output_dir: str,
    parallel: bool,
    workers: int | None,
    data_file: str | None,
) -> None:
    """Execute an optimization in a background thread."""
    import json as _json
    import os

    import numpy as np

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

    os.makedirs(output_dir, exist_ok=True)

    _update_job(app, job_id, progress="Loading model...")
    desc = load_model_description(model_file)
    expt_cfg = {**EXPERIMENT_DEFAULTS, **desc.get("experiment", {})}
    opt_cfg = {**OPTIMIZATION_DEFAULTS, **desc.get("optimization", {})}

    param = opt_cfg["param"]
    param_vals = [float(v) for v in opt_cfg["param_values"]]
    poi = opt_cfg.get("parameters_of_interest")
    if isinstance(poi, list):
        poi = [str(s) for s in poi]
    num_realizations = int(opt_cfg["num_realizations"])
    mcmc_steps = int(opt_cfg["mcmc_steps"])
    entropy_method = str(opt_cfg["entropy_method"])

    effective_data_file = data_file or expt_cfg.get("data_file")

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

    experiment = load_experiment(model_file, simulator.q_values, simulator.dq_values)
    designer = ExperimentDesigner(
        experiment, simulator=simulator, parameters_of_interest=poi
    )
    h_prior = designer.prior_entropy()

    _update_job(
        app,
        job_id,
        progress=f"Optimizing {param} ({len(param_vals)} values, "
        f"{num_realizations} realizations)...",
    )

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

    best_idx = int(np.argmax([r[1] for r in results]))
    best_val, best_gain, best_std = results[best_idx]

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
        _json.dump(result_dict, f, indent=2)

    _update_job(app, job_id, progress="Generating plots...")
    make_report(json_file=json_path, output_dir=output_dir)

    # Copy model file to output dir for reference
    import shutil

    model_dest = os.path.join(output_dir, Path(model_file).name)
    if not os.path.exists(model_dest):
        shutil.copy2(model_file, model_dest)

    _update_job(
        app,
        job_id,
        status="complete",
        progress=f"Complete — optimal {param} = {best_val:.4g} "
        f"(ΔH = {best_gain:.3f} bits)",
        result_dir=Path(output_dir).name,
    )

    # Update RESULTS_DIR so the results page shows the new run
    results_parent = str(Path(output_dir).parent)
    app.config["RESULTS_DIR"] = results_parent


def _run_plan_job(
    app,
    job_id: str,
    description: str,
    output_dir: str,
    parallel: bool,
    workers: int | None,
    data_file: str | None,
) -> None:
    """Execute plan → optimize in a background thread."""
    import os

    from dotenv import load_dotenv

    load_dotenv()

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate model via LLM
    _update_job(app, job_id, progress="Generating model via LLM...")

    from rose.modeler.llm_generator import generate_model_yaml

    # Read LLM config from environment
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    model_name = os.environ.get("LLM_MODEL", "gpt-4o")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or None
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

    # For ALCF, use the access token as the API key
    if provider == "alcf" and not api_key:
        api_key = os.environ.get("ALCF_ACCESS_TOKEN")

    yaml_text = generate_model_yaml(
        description,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
    )

    model_path = os.path.join(output_dir, "generated_model.yaml")
    with open(model_path, "w") as f:
        f.write(yaml_text)

    _update_job(app, job_id, progress="Model generated. Running optimization...")

    # Step 2: Optimize
    _run_optimize_job(app, job_id, model_path, output_dir, parallel, workers, data_file)
