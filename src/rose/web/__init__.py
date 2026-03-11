"""
Flask web application for visualizing ROSE optimization results.

Usage::

    from rose.web import create_app
    app = create_app("/path/to/results")
    app.run(port=5000)

Or via the CLI::

    rose serve results/
    rose serve results/ --port 8080
"""

from __future__ import annotations

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
    app.secret_key = "rose-web"
    app.register_blueprint(bp)
    return app
