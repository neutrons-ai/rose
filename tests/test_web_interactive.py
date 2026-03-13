"""Tests for Phase 4 interactive web features."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def app(tmp_path):
    """Create a Phase 4 test app."""
    from rose.web import create_app

    return create_app(str(tmp_path))


@pytest.fixture()
def client(app):
    return app.test_client()


# ── Interactive page routes ──────────────────────────────────────


class TestInteractivePages:
    """Tests for the new optimize and plan page routes."""

    def test_optimize_page_returns_200(self, client):
        resp = client.get("/optimize")
        assert resp.status_code == 200
        assert b"New Optimization" in resp.data

    def test_optimize_page_has_form(self, client):
        resp = client.get("/optimize")
        assert b"model-file" in resp.data
        assert b"output-dir" in resp.data
        assert b"Start Optimization" in resp.data

    def test_plan_page_returns_200(self, client):
        resp = client.get("/plan")
        assert resp.status_code == 200
        assert b"Plan" in resp.data

    def test_plan_page_has_description_field(self, client):
        resp = client.get("/plan")
        assert b"sample-desc" in resp.data
        assert b"Generate Model" in resp.data

    def test_nav_has_optimize_tab(self, client):
        resp = client.get("/")
        assert b"Optimize" in resp.data
        assert b"Plan" in resp.data


# ── File browser API ─────────────────────────────────────────────


class TestBrowseAPI:
    """Tests for the server-side file/folder browser."""

    def test_browse_files_home(self, client):
        resp = client.get("/api/browse-files")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current" in data
        assert "entries" in data
        assert isinstance(data["entries"], list)

    def test_browse_files_with_path(self, client, tmp_path):
        # Create test files
        (tmp_path / "test.yaml").write_text("name: test")
        (tmp_path / "sub").mkdir()

        resp = client.get(f"/api/browse-files?path={tmp_path}")
        data = resp.get_json()
        assert resp.status_code == 200
        names = [e["name"] for e in data["entries"]]
        assert "sub" in names

    def test_browse_files_ext_filter(self, client, tmp_path):
        (tmp_path / "model.yaml").write_text("name: test")
        (tmp_path / "data.txt").write_text("1 2 3 4")

        resp = client.get(f"/api/browse-files?path={tmp_path}&ext=.yaml")
        data = resp.get_json()
        file_names = [e["name"] for e in data["entries"] if not e.get("is_dir")]
        assert "model.yaml" in file_names
        assert "data.txt" not in file_names

    def test_browse_files_nonexistent(self, client):
        resp = client.get("/api/browse-files?path=/nonexistent_path_xyz")
        assert resp.status_code == 400

    def test_browse_dirs(self, client, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("hi")

        resp = client.get(f"/api/browse-dirs?path={tmp_path}")
        data = resp.get_json()
        assert resp.status_code == 200
        names = [e["name"] for e in data["entries"]]
        assert "subdir" in names
        # Files should not appear in dir listing
        assert "file.txt" not in names

    def test_browse_dirs_has_parent(self, client, tmp_path):
        resp = client.get(f"/api/browse-dirs?path={tmp_path}")
        data = resp.get_json()
        assert data["parent"] is not None

    def test_hidden_files_excluded(self, client, tmp_path):
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.yaml").write_text("name: test")

        resp = client.get(f"/api/browse-files?path={tmp_path}")
        data = resp.get_json()
        names = [e["name"] for e in data["entries"]]
        assert ".hidden" not in names
        assert "visible.yaml" in names

    def test_browse_files_path_traversal(self, client):
        """Path traversal via .. in browse API should not crash."""
        resp = client.get("/api/browse-files?path=/../../../etc")
        # Resolved path either exists (200 on macOS/Linux) or not (400)
        assert resp.status_code in (200, 400)

    def test_browse_files_fallback_file_to_parent(self, client, tmp_path):
        """When path points to a file (not dir), browse falls back to parent."""
        f = tmp_path / "a_file.txt"
        f.write_text("hello")

        resp = client.get(f"/api/browse-files?path={f}")
        assert resp.status_code == 200
        data = resp.get_json()
        # Should have fallen back to the parent directory
        assert Path(data["current"]) == tmp_path.resolve()

    def test_browse_files_permission_denied(self, client, tmp_path):
        """Restricted directory returns 403."""
        restricted = tmp_path / "noaccess"
        restricted.mkdir()
        restricted.chmod(0o000)
        try:
            resp = client.get(f"/api/browse-files?path={restricted}")
            assert resp.status_code == 403
        finally:
            restricted.chmod(0o755)


# ── Job API ──────────────────────────────────────────────────────


class TestJobAPI:
    """Tests for the background job management API."""

    def test_optimize_job_validation_no_model(self, client):
        resp = client.post(
            "/api/jobs/optimize",
            json={"model_file": "", "output_dir": "/tmp"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert any("model_file" in e for e in data["errors"])

    def test_optimize_job_validation_no_output(self, client):
        resp = client.post(
            "/api/jobs/optimize",
            json={"model_file": "/etc/hosts", "output_dir": ""},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert any("output_dir" in e for e in data["errors"])

    def test_plan_job_validation_short_desc(self, client):
        resp = client.post(
            "/api/jobs/plan",
            json={"description": "short", "output_dir": "/tmp"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert any("description" in e for e in data["errors"])

    def test_plan_job_validation_no_output(self, client):
        resp = client.post(
            "/api/jobs/plan",
            json={"description": "A sufficiently long description", "output_dir": ""},
        )
        assert resp.status_code == 400

    def test_job_status_not_found(self, client):
        resp = client.get("/api/jobs/nonexistent/status")
        assert resp.status_code == 404

    def test_job_status_returns_state(self, app, client):
        """Insert a fake job and verify status endpoint returns it."""
        with app.config["JOBS_LOCK"]:
            app.config["JOBS"]["test-123"] = {
                "id": "test-123",
                "type": "optimize",
                "status": "running",
                "progress": "Working...",
                "error": None,
            }

        resp = client.get("/api/jobs/test-123/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == "test-123"
        assert data["status"] == "running"
        assert data["progress"] == "Working..."

    def test_job_status_hides_internal_keys(self, app, client):
        """Keys starting with _ should not appear in status response."""
        with app.config["JOBS_LOCK"]:
            app.config["JOBS"]["test-priv"] = {
                "id": "test-priv",
                "type": "optimize",
                "status": "running",
                "progress": "...",
                "_internal_thread": "hidden",
            }

        resp = client.get("/api/jobs/test-priv/status")
        data = resp.get_json()
        assert "_internal_thread" not in data

    def test_optimize_job_bad_data_file(self, client, tmp_path):
        model = tmp_path / "model.yaml"
        model.write_text("layers: []")

        resp = client.post(
            "/api/jobs/optimize",
            json={
                "model_file": str(model),
                "output_dir": str(tmp_path / "out"),
                "data_file": "/nonexistent_file.txt",
            },
        )
        assert resp.status_code == 400
        assert any("data_file" in e for e in resp.get_json()["errors"])

    def test_optimize_job_accepted(self, app, client, tmp_path):
        """Valid optimize request returns 200 with job_id and creates job entry."""
        model = tmp_path / "model.yaml"
        model.write_text("layers: []")

        resp = client.post(
            "/api/jobs/optimize",
            json={
                "model_file": str(model),
                "output_dir": str(tmp_path / "out"),
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data
        assert data["status"] == "started"
        # Verify job exists in app config
        with app.config["JOBS_LOCK"]:
            job = app.config["JOBS"][data["job_id"]]
            assert job["type"] == "optimize"
            assert job["status"] in ("running", "error")  # thread may fail fast

    def test_plan_job_accepted(self, app, client, tmp_path):
        """Valid plan request returns 200 with job_id and creates job entry."""
        resp = client.post(
            "/api/jobs/plan",
            json={
                "description": "A 50 nm Cu film deposited on silicon wafer, measured in air",
                "output_dir": str(tmp_path / "out"),
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data
        assert data["status"] == "started"
        with app.config["JOBS_LOCK"]:
            job = app.config["JOBS"][data["job_id"]]
            assert job["type"] == "plan"


# ── AuRE plugin registration ────────────────────────────────────


class TestPluginRegistration:
    """Tests for mounting ROSE as an AuRE plugin blueprint."""

    def test_register_with_aure(self):
        from flask import Flask

        from rose.web import register_with_aure

        aure_app = Flask(__name__)
        aure_app.secret_key = "test"
        register_with_aure(aure_app)

        with aure_app.test_client() as c:
            assert c.get("/rose/").status_code == 200
            assert c.get("/rose/optimize").status_code == 200
            assert c.get("/rose/plan").status_code == 200
            assert c.get("/rose/api/results").status_code == 200

    def test_register_with_custom_prefix(self):
        from flask import Flask

        from rose.web import register_with_aure

        app = Flask(__name__)
        app.secret_key = "test"
        register_with_aure(app, url_prefix="/tools/rose")

        with app.test_client() as c:
            assert c.get("/tools/rose/optimize").status_code == 200

    def test_register_adds_jobs_config(self):
        from flask import Flask

        from rose.web import register_with_aure

        app = Flask(__name__)
        app.secret_key = "test"
        register_with_aure(app)

        assert "JOBS" in app.config
        assert "JOBS_LOCK" in app.config
        assert hasattr(app.config["JOBS_LOCK"], "acquire")
