"""Flask blueprint – page routes and JSON API endpoints for ROSE."""

from __future__ import annotations

from pathlib import Path

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
)

from .data import ResultData, list_results

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
