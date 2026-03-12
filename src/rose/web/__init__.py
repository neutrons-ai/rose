"""
Flask web application for visualizing ROSE optimization results.

Usage::

    from rose.web import create_app
    app = create_app("/path/to/results")
    app.run(port=5000)

Or via the CLI::

    rose serve results/
    rose serve results/ --port 8080

AuRE plugin usage::

    from rose.web import register_with_aure
    register_with_aure(aure_app)
"""

from __future__ import annotations

import threading

from flask import Flask

from .routes import bp


def create_app(results_dir: str | None = None) -> Flask:
    """Create the Flask application.

    Args:
        results_dir: Path to the directory containing optimization
            result sub-directories.  Each sub-directory should have
            an ``optimization_results.json`` file.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)
    app.config["RESULTS_DIR"] = results_dir
    app.config["JOBS"] = {}  # job_id → job state dict
    app.config["JOBS_LOCK"] = threading.Lock()
    app.secret_key = "rose-web"
    app.register_blueprint(bp)
    return app


def register_with_aure(app: Flask, url_prefix: str = "/rose") -> None:
    """Register the ROSE blueprint with an existing Flask app (e.g. AuRE).

    Args:
        app: The Flask application to mount ROSE onto.
        url_prefix: URL prefix for all ROSE routes (default ``/rose``).
    """
    if "JOBS" not in app.config:
        app.config["JOBS"] = {}
    if "JOBS_LOCK" not in app.config:
        app.config["JOBS_LOCK"] = threading.Lock()
    app.register_blueprint(bp, url_prefix=url_prefix)
