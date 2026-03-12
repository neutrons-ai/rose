/* ================================================================
   ROSE – setup.js
   Client-side logic for the interactive setup pages:
     • Server-side file / folder browser (modal)
     • Job launch + status polling
     • localStorage persistence of form values
   ================================================================ */

/* ---- state --------------------------------------------------- */
let browserMode = "file";        // "file" | "dir" | "datafile"
let browserCurrentPath = null;
let browserParentPath = null;
let browserModalInstance = null;
let pollTimer = null;
let currentJobId = null;

const STORAGE_KEY = "rose_setup";

/* ---- persist / restore form values --------------------------- */

function _saveFormValues() {
  const vals = {};
  const fields = ["model-file", "data-file", "output-dir", "sample-desc"];
  fields.forEach(function (id) {
    const el = document.getElementById(id);
    if (el) vals[id] = el.value;
  });
  const par = document.getElementById("run-parallel");
  if (par) vals["run-parallel"] = par.checked;
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(vals)); } catch (_) {}
}

function _restoreFormValues() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const vals = JSON.parse(raw);
    Object.keys(vals).forEach(function (id) {
      const el = document.getElementById(id);
      if (!el) return;
      if (id === "run-parallel") {
        el.checked = vals[id];
      } else {
        el.value = vals[id];
      }
    });
  } catch (_) {}
}

document.addEventListener("DOMContentLoaded", _restoreFormValues);

/* ---- file / folder browser ----------------------------------- */

function openBrowser(mode) {
  browserMode = mode;

  const titles = {
    file: "Select Model File",
    datafile: "Select Data File",
    dir: "Select Output Folder",
  };
  document.getElementById("browser-title").textContent = titles[mode] || "Browse";

  // Show "Select this folder" button only in dir mode
  document.getElementById("btn-select").style.display =
    mode === "dir" ? "inline-block" : "none";

  _fetchBrowserListing("");

  browserModalInstance =
    browserModalInstance ||
    new bootstrap.Modal(document.getElementById("browserModal"));
  browserModalInstance.show();
}

function _fetchBrowserListing(path) {
  const isDir = browserMode === "dir";
  const endpoint = isDir ? "/api/browse-dirs" : "/api/browse-files";
  const params = new URLSearchParams();
  if (path) params.set("path", path);

  // Extension filter for model files vs data files
  if (browserMode === "file") params.set("ext", ".yaml");
  if (browserMode === "datafile") params.set("ext", ".txt");

  fetch(`${endpoint}?${params}`)
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.error) { alert(data.error); return; }

      browserCurrentPath = data.current;
      browserParentPath = data.parent;
      document.getElementById("browser-path").textContent = data.current;
      document.getElementById("btn-parent").disabled = !data.parent;

      const list = document.getElementById("browser-list");
      list.innerHTML = "";

      data.entries.forEach(function (entry) {
        const a = document.createElement("a");
        a.className =
          "list-group-item list-group-item-action d-flex align-items-center";
        a.href = "#";

        const icon = document.createElement("i");
        icon.className =
          (entry.is_dir || entry.is_dir === undefined)
            ? "bi bi-folder-fill text-warning me-2"
            : "bi bi-file-earmark-text me-2";
        a.appendChild(icon);

        const name = document.createElement("span");
        name.textContent = entry.name;
        a.appendChild(name);

        a.addEventListener("click", function (e) {
          e.preventDefault();
          if (entry.is_dir || entry.is_dir === undefined) {
            _fetchBrowserListing(entry.path);
          } else {
            // File selected
            const targetId =
              browserMode === "datafile" ? "data-file" : "model-file";
            document.getElementById(targetId).value = entry.path;
            browserModalInstance.hide();
          }
        });

        list.appendChild(a);
      });

      if (data.entries.length === 0) {
        const empty = document.createElement("div");
        empty.className = "list-group-item text-muted text-center";
        empty.textContent = isDir
          ? "No sub-folders"
          : "No matching files in this directory";
        list.appendChild(empty);
      }
    })
    .catch(function (err) { console.error("Browse error:", err); });
}

function browserUp() {
  if (browserParentPath) _fetchBrowserListing(browserParentPath);
}

function browserSelect() {
  if (browserMode === "dir" && browserCurrentPath) {
    document.getElementById("output-dir").value = browserCurrentPath;
    browserModalInstance.hide();
  }
}

/* ---- optimization launch ------------------------------------- */

function startOptimize() {
  const modelFile = document.getElementById("model-file").value.trim();
  const outputDir = document.getElementById("output-dir").value.trim();

  if (!modelFile) { alert("Please select a model file."); return; }
  if (!outputDir) { alert("Please select an output directory."); return; }

  _saveFormValues();
  _disableStartButton();

  const dataFileEl = document.getElementById("data-file");
  const dataFile = dataFileEl ? dataFileEl.value.trim() : "";

  fetch("/api/jobs/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_file: modelFile,
      output_dir: outputDir,
      parallel: document.getElementById("run-parallel").checked,
      data_file: dataFile || null,
    }),
  })
    .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
    .then(function (res) {
      if (!res.ok) {
        const msg = res.data.errors
          ? res.data.errors.join("\n")
          : res.data.error || "Unknown error";
        alert("Could not start optimization:\n" + msg);
        _enableStartButton();
        return;
      }
      _switchToProgress(res.data.job_id);
    })
    .catch(function (err) {
      alert("Network error: " + err);
      _enableStartButton();
    });
}

/* ---- plan launch --------------------------------------------- */

function startPlan() {
  const desc = document.getElementById("sample-desc").value.trim();
  const outputDir = document.getElementById("output-dir").value.trim();

  if (!desc || desc.length < 10) {
    alert("Please enter a sample description (at least 10 characters).");
    return;
  }
  if (!outputDir) { alert("Please select an output directory."); return; }

  _saveFormValues();
  _disableStartButton();

  const dataFileEl = document.getElementById("data-file");
  const dataFile = dataFileEl ? dataFileEl.value.trim() : "";

  fetch("/api/jobs/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      description: desc,
      output_dir: outputDir,
      parallel: document.getElementById("run-parallel").checked,
      data_file: dataFile || null,
    }),
  })
    .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
    .then(function (res) {
      if (!res.ok) {
        const msg = res.data.errors
          ? res.data.errors.join("\n")
          : res.data.error || "Unknown error";
        alert("Could not start plan:\n" + msg);
        _enableStartButton();
        return;
      }
      _switchToProgress(res.data.job_id);
    })
    .catch(function (err) {
      alert("Network error: " + err);
      _enableStartButton();
    });
}

/* ---- status polling ------------------------------------------ */

function _switchToProgress(jobId) {
  currentJobId = jobId;
  document.getElementById("setup-section").style.display = "none";
  document.getElementById("progress-section").style.display = "";
  _pollStatus();
}

function _pollStatus() {
  if (pollTimer) clearTimeout(pollTimer);
  if (!currentJobId) return;

  fetch(`/api/jobs/${currentJobId}/status`)
    .then(function (r) { return r.json(); })
    .then(function (st) {
      document.getElementById("progress-text").textContent =
        st.progress || st.status;

      // Status badge
      const badge = document.getElementById("progress-status");
      badge.textContent = st.status;
      badge.className = "badge " + (
        st.status === "complete" ? "bg-success" :
        st.status === "error" ? "bg-danger" : "bg-primary"
      );

      // Progress bar
      const bar = document.getElementById("progress-bar");
      if (st.status === "complete") {
        bar.style.width = "100%";
        bar.classList.remove("progress-bar-animated", "progress-bar-striped");
        bar.classList.add("bg-success");
      } else if (st.status === "error") {
        bar.classList.remove("progress-bar-animated", "progress-bar-striped");
        bar.classList.add("bg-danger");
        document.getElementById("progress-text").textContent =
          "Error: " + (st.error || "unknown");
      } else {
        // Indeterminate — pulse between 10–90%
        const cur = parseInt(bar.style.width, 10) || 10;
        bar.style.width = Math.min(90, cur + 5) + "%";
      }

      // Show footer on completion / error
      if (st.status === "complete" || st.status === "error") {
        document.getElementById("progress-footer").style.display = "";
        // Update results link
        if (st.result_dir) {
          document.getElementById("btn-view-results").href =
            "/results/" + st.result_dir;
        }
        return; // stop polling
      }

      pollTimer = setTimeout(_pollStatus, 2000);
    })
    .catch(function () {
      pollTimer = setTimeout(_pollStatus, 3000);
    });
}

/* ---- UI helpers ---------------------------------------------- */

function _disableStartButton() {
  const btn = document.getElementById("btn-start");
  btn.disabled = true;
  btn.innerHTML =
    '<span class="spinner-border spinner-border-sm"></span> Starting…';
}

function _enableStartButton() {
  const btn = document.getElementById("btn-start");
  btn.disabled = false;
  // Restore appropriate label
  if (btn.textContent.includes("Generate")) {
    btn.innerHTML = '<i class="bi bi-magic"></i> Generate Model &amp; Optimize';
  } else {
    btn.innerHTML = '<i class="bi bi-play-fill"></i> Start Optimization';
  }
}

function resetSetup() {
  document.getElementById("setup-section").style.display = "";
  document.getElementById("progress-section").style.display = "none";
  document.getElementById("progress-footer").style.display = "none";

  const bar = document.getElementById("progress-bar");
  bar.style.width = "10%";
  bar.className = "progress-bar progress-bar-striped progress-bar-animated";

  _enableStartButton();
  currentJobId = null;
}
