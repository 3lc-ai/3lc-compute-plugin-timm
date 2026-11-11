
// timm UI — SocketIO handlers, form logic, config management

// ── State ─────────────────────────────────────────────────────
var tmSocket = null;
var tmPresets = null;           // the SDK's preset bar (_tlcPluginPresets), once mounted
var _tmInitialParams = null;    // the parameter fields as the page loaded: Default settings
var tmModelsCache = null;       // [{name, family, popular}]
var tmDetectedTask = null;
var tmDetectedImageCol = "";
var tmDetectedLabelCol = "";
var tmCurrentMode = "train";
var _tmCurrentJobId = null;
var tmDetecting = false;        // the train table is being read (from the change, through the debounce)
var tmTaskError = "";           // why the train table could not be read; "" when it was, or is not yet
var _tmSubmitting = false;      // Start was pressed and the run request has not answered yet

var _TM_STORAGE_KEY = "timm_selected_config";  // the old config bar's key; the preset bar honours it once

var TM_COMPUTE_URL = PLUGIN_API.getConfig("compute_service_url");

// ── Helpers ───────────────────────────────────────────────────
function _tmFetch(path, opts) {
  opts = opts || {};
  // The venv worker can be cold on first use — it imports torch and builds the
  // timm model registry on startup, which can exceed authFetch's default 10s
  // abort and surface as "AbortError: signal is aborted without reason". Give
  // worker calls a more generous default; callers may still override per-request.
  if (opts.timeout === undefined) opts.timeout = 60000;
  return PLUGIN_API.authFetch(TM_COMPUTE_URL + path, opts);
}

function _tmFetchJson(path, opts) {
  return _tmFetch(path, opts).then(function(r) { return r.json(); });
}

function _tmPost(path, body) {
  return _tmFetchJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function _esc(s) { var d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

function _tmSetTaskBadge(task, detail) {
  var badge = document.getElementById("tm-task-badge");
  var detailEl = document.getElementById("tm-task-detail");
  badge.textContent = task === "error" ? "error" : task.replace(/_/g, " ");
  badge.className = "tm-task-badge-" + (task === "error" ? "unknown" : task);
  detailEl.textContent = detail || "";
}

function _tmFmtDuration(sec) {
  if (sec < 60) return Math.round(sec) + "s";
  if (sec < 3600) return Math.floor(sec / 60) + "m " + Math.round(sec % 60) + "s";
  return Math.floor(sec / 3600) + "h " + Math.floor((sec % 3600) / 60) + "m";
}

// ── SocketIO Connection ──────────────────────────────────────
function tmInitSocket() {
  var io = PLUGIN_API.libs.io;
  if (!io) { tmLog("SocketIO not available"); return; }

  tmSocket = io(TM_COMPUTE_URL + "/timm", {
    transports: ["websocket", "polling"],
    reconnection: true,
    reconnectionDelay: 2000,
  });

  tmSocket.on("connect", function() { tmLog("Connected to Compute Service"); });
  tmSocket.on("disconnect", function() { tmLog("Disconnected"); });
  tmSocket.on("connect_error", function() {});

  tmSocket.on("epoch_progress", function(data) {
    tmShowCurrentJob(data.job_id);
    var m = data.metrics || {};
    var batch = m.batch || 0;
    var totalBatches = m.total_batches || 0;
    var phase = m.phase || "";
    var timing = data.timing || {};

    var epochFraction = totalBatches > 0 ? (batch / totalBatches) : 1;
    var overallPct = ((data.epoch - 1 + epochFraction) / data.total_epochs) * 100;
    overallPct = Math.min(overallPct, 100);
    document.getElementById("tm-epoch-bar").style.width = overallPct.toFixed(1) + "%";
    document.getElementById("tm-pct-label").textContent = Math.round(overallPct) + "%";

    if (batch && totalBatches && phase) {
      var phaseLabel = phase === "val" ? "Val" : "Train";
      document.getElementById("tm-epoch-label").textContent =
        "Epoch " + data.epoch + "/" + data.total_epochs + " · " + phaseLabel + " batch " + batch + "/" + totalBatches;
    } else {
      document.getElementById("tm-epoch-label").textContent =
        "Epoch " + data.epoch + "/" + data.total_epochs;
    }

    var parts = [];
    if (timing.elapsed_s !== undefined) parts.push("Elapsed: " + _tmFmtDuration(timing.elapsed_s));
    if (timing.eta_s !== undefined) parts.push("ETA: " + _tmFmtDuration(timing.eta_s));
    if (timing.avg_epoch_s !== undefined) parts.push("Per epoch: " + _tmFmtDuration(timing.avg_epoch_s));
    document.getElementById("tm-timing").innerHTML = parts.join(' &nbsp;|&nbsp; ');

    tmRenderMetrics(m);
  });

  tmSocket.on("job_status", function(data) {
    tmShowCurrentJob(data.job_id);
    tmLog("[" + data.job_id.slice(0, 8) + "] " + data.status + ": " + data.message);
    // Show processing badge during collection phase
    if (data.status === "collecting" || (data.message && data.message.indexOf("Collecting") === 0)) {
      var metricsEl = document.getElementById("tm-current-metrics");
      if (metricsEl) {
        metricsEl.innerHTML = '<span class="plugin-status-running" style="display:inline-block;padding:3px 10px;border-radius:3px;font-size:10px;font-weight:600">' +
          _esc(data.message || "Collecting metrics...") + '</span>';
      }
      document.getElementById("tm-epoch-label").textContent = "Collecting metrics";
      document.getElementById("tm-epoch-bar").style.width = "100%";
      document.getElementById("tm-pct-label").textContent = "";
    }
  });

  tmSocket.on("queue_update", function(data) {
    tmRenderQueue(data);
  });
}

// ── Generic job lifecycle (host job_update channel) ─────────────────────
// Completion and failure are NOT plugin events: the plugin reports the run URL via
// ctx.result and failures by raising, and the host's terminal job_update carries
// run_url / error. The SDK's PluginJobs client (auto-injected by the host) tracks it.
var _tmUntrack = null;

function _tmTrackJob(jobId) {
  if (!jobId || typeof PluginJobs === "undefined") return;
  if (_tmUntrack) { _tmUntrack(); _tmUntrack = null; }
  _tmUntrack = PluginJobs.track("/timm", jobId, {
    onUpdate: function(job) { _tmRenderLiveBanner(job); },
    onDone: function(job) {
      _tmUntrack = null;
      var status = job.status || "completed";
      if (status === "completed") {
        tmLog("Job completed." + (job.run_url ? " Run: " + job.run_url : ""));
      } else {
        tmLog("Job " + status + ".");
      }
      tmFinishCurrentJob(status, job);
    },
    onError: function(job) {
      _tmUntrack = null;
      tmLog("Job FAILED: " + (job.error || job.subtitle || "unknown error"));
      tmFinishCurrentJob("failed", job);
    },
  });
}

// Where the Open button goes for a generic job record: the run detail page when the
// job reported a run URL, else the project page when one is known.
function _tmOpenHref(job, projectName) {
  if (job && job.run_url) return "/runs/" + encodeURIComponent(job.run_url);
  if (projectName) return "/projects/" + encodeURIComponent(projectName);
  return "";
}

// Compact "job running — watch it in the Queue" state above the launch form. Fed by
// the seed-on-mount list call and every job_update while the job is live; hidden
// again once the job reaches a terminal state (the Queue panel is the durable view).
function _tmRenderLiveBanner(job) {
  var banner = document.getElementById("tm-live-banner");
  if (!banner || !job) return;
  var live = job.status === "queued" || job.status === "running";
  if (!live) { banner.style.display = "none"; return; }
  banner.style.display = "flex";
  var statusEl = document.getElementById("tm-live-status");
  statusEl.textContent = job.status;
  statusEl.className = "tm-queue-status plugin-status-" + job.status;
  document.getElementById("tm-live-title").textContent = job.title || "timm job";
  var p = job.progress || {};
  var detail = [];
  if (p.label) detail.push(p.label);
  if (typeof p.percent === "number" && p.percent >= 0) detail.push(Math.round(p.percent) + "%");
  var t = p.timing || {};
  if (t.eta_s !== undefined && t.eta_s !== null) detail.push("ETA " + _tmFmtDuration(t.eta_s));
  document.getElementById("tm-live-detail").textContent = detail.join(" · ");
  document.getElementById("tm-live-error").style.display = "none";
  var openEl = document.getElementById("tm-live-open");
  var href = _tmOpenHref(job, null);
  openEl.style.display = href ? "" : "none";
  if (href) openEl.href = href;
}

// Seed on mount: the fragment is torn down on navigation and job_update is live-only,
// so ask the host which timm jobs are already queued/running and render their state
// instead of an empty form over a live job.
function _tmSeedFromHost() {
  if (typeof PluginJobs === "undefined" || !PluginJobs.list) return;
  PluginJobs.list("timm").then(function(jobs) {
    var live = (jobs || []).filter(function(j) { return j.status === "queued" || j.status === "running"; });
    if (!live.length) return;
    var job = live[0];
    _tmRenderLiveBanner(job);
    tmShowCurrentJob(job.id, job.title || "timm job");
    var statusEl = document.getElementById("tm-current-status");
    statusEl.textContent = job.status;
    statusEl.className = "tm-queue-status plugin-status-" + job.status;
    var p = job.progress || {};
    if (typeof p.percent === "number" && p.percent >= 0) {
      document.getElementById("tm-epoch-bar").style.width = p.percent.toFixed(1) + "%";
      document.getElementById("tm-pct-label").textContent = Math.round(p.percent) + "%";
    }
    if (p.label) document.getElementById("tm-epoch-label").textContent = p.label;
    tmLog("Resumed tracking " + job.status + " job " + String(job.id).slice(0, 8));
    _tmTrackJob(job.id);
  }).catch(function(e) { tmLog("Could not list jobs: " + e); });
}

// ── Logging ──────────────────────────────────────────────────
function tmLog(msg) {
  var el = document.getElementById("tm-log");
  if (!el) return;
  var line = document.createElement("div");
  line.textContent = "[" + new Date().toLocaleTimeString() + "] " + msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

// ── Mode Toggle ─────────────────────────────────────────────
function tmSetMode(mode) {
  tmCurrentMode = mode;
  var trainBtn = document.getElementById("tm-mode-train");
  var collectBtn = document.getElementById("tm-mode-collect");
  var helpEl = document.getElementById("tm-mode-help");
  var startBtn = document.getElementById("tm-train-btn");
  var trainParams = document.getElementById("tm-train-params");

  if (mode === "collect") {
    trainBtn.className = "btn btn-sm";
    trainBtn.setAttribute("aria-pressed", "false");
    collectBtn.className = "btn btn-sm active";
    collectBtn.setAttribute("aria-pressed", "true");
    helpEl.textContent = "Run inference and collect embeddings + predictions (no training).";
    trainParams.style.display = "none";
  } else {
    trainBtn.className = "btn btn-sm active";
    trainBtn.setAttribute("aria-pressed", "true");
    collectBtn.className = "btn btn-sm";
    collectBtn.setAttribute("aria-pressed", "false");
    helpEl.textContent = "Train a model with 3LC metrics collection.";
    trainParams.style.display = "";
  }
  startBtn.textContent = tmStartLabel();
  tmGate();
}

// ── Start: disabled until it can run, saying why (PBI 19074) ──
// Start's label follows the mode; the reasons name it as the button shows it.
function tmStartLabel() {
  return tmCurrentMode === "collect" ? "Start Collection" : "Start timm Training";
}

// Why Start cannot run yet, as what to do to enable it; "" when it can.
function tmStartBlocker() {
  var label = tmStartLabel();
  if (!document.getElementById("tm-train-url").value.trim()) return "Choose a train table to enable " + label + ".";
  if (tmDetecting) return "Wait while the train table is read to enable " + label + ".";
  if (tmTaskError) return "Choose another train table to enable " + label + ": this one could not be read.";
  // "unknown" is a table without an image column (task_detection.py): nothing to train or collect on.
  if (tmDetectedTask === "unknown") return "Choose a train table with an image column to enable " + label + ".";
  // Any model name timm can create is fine, listed or not: an untagged name like resnet50 (timm picks its
  // default weights) or an architecture fine-tuned from a Pretrained Model URL. The list is only a picker.
  if (!document.getElementById("tm-model-search").value.trim()) return "Choose a model to enable " + label + ".";
  if (typeof PluginJobs === "undefined") return "Reload the page to enable " + label + ".";
  return "";
}

// Start, disabled while it cannot run, with the reason in its tooltip and on the line under it;
// Swap, disabled while both table fields are empty, with the reason in its tooltip.
function tmGate() {
  var btn = document.getElementById("tm-train-btn");
  var reason = tmStartBlocker();
  btn.disabled = _tmSubmitting || !!reason;
  btn.title = reason;
  document.getElementById("tm-train-why").textContent = reason;  // the line keeps its height when empty
  var swap = document.getElementById("tm-swap-btn");
  var noTables = !document.getElementById("tm-train-url").value.trim() &&
    !document.getElementById("tm-val-url").value.trim();
  swap.disabled = noTables;
  swap.title = noTables ? "Nothing to swap: both table fields are empty." : swap.dataset.title;
}

function _tmEnforceTaskMode(task) {
  var trainBtn = document.getElementById("tm-mode-train");
  if (task === "feature_extraction") {
    tmSetMode("collect");
    trainBtn.disabled = true;
    trainBtn.title = "Choose a train table with a label column to enable Train.";
  } else {
    trainBtn.disabled = false;
    trainBtn.title = "";
  }
}

// ── Model Search ─────────────────────────────────────────────
function tmLoadModels() {
  if (tmModelsCache !== null) return;
  _tmFetchJson("/api/plugins/timm/models")
    .then(function(data) {
      tmModelsCache = Array.isArray(data) ? data : [];
      tmLog("Loaded " + tmModelsCache.length + " timm models");
    })
    .catch(function(e) { tmLog("Failed to load models: " + e); });
}

function tmOpenModelDropdown() {
  if (!tmModelsCache) { tmLoadModels(); return; }
  tmFilterModels();
}

function tmFilterModels() {
  var dd = document.getElementById("tm-model-dropdown");
  var query = document.getElementById("tm-model-search").value.trim().toLowerCase();

  if (!tmModelsCache) { dd.classList.remove("open"); return; }

  var filtered = tmModelsCache;
  if (query) {
    filtered = tmModelsCache.filter(function(m) {
      return m.name.toLowerCase().indexOf(query) !== -1 || m.family.toLowerCase().indexOf(query) !== -1;
    });
  }

  // Limit display
  var popular = filtered.filter(function(m) { return m.popular; });
  var others = filtered.filter(function(m) { return !m.popular; });
  var maxOthers = query ? 50 : 20;
  if (others.length > maxOthers) others = others.slice(0, maxOthers);

  dd.innerHTML = "";

  if (popular.length > 0 && !query) {
    var label = document.createElement("div");
    label.className = "tm-model-family-label";
    label.textContent = "POPULAR";
    dd.appendChild(label);
    popular.forEach(function(m) { dd.appendChild(_tmModelItem(m, true)); });
  }

  // Group others by family
  var families = {};
  (query ? filtered : others).forEach(function(m) {
    if (!query && m.popular) return;
    var f = m.family;
    if (!families[f]) families[f] = [];
    families[f].push(m);
  });

  Object.keys(families).sort().forEach(function(f) {
    var label = document.createElement("div");
    label.className = "tm-model-family-label";
    label.textContent = f.toUpperCase();
    dd.appendChild(label);
    families[f].forEach(function(m) { dd.appendChild(_tmModelItem(m, false)); });
  });

  dd.classList.add("open");
}

function _tmModelItem(m, isPopular) {
  var div = document.createElement("div");
  var selected = document.getElementById("tm-model-search").value === m.name;
  div.className = "tm-model-item" + (isPopular ? " tm-model-popular" : "") + (selected ? " tm-model-selected" : "");

  var nameSpan = document.createElement("span");
  nameSpan.className = "tm-model-name";
  nameSpan.textContent = m.name;
  div.appendChild(nameSpan);

  // Meta info (image size, dataset)
  var parts = [];
  if (m.image_size) parts.push(m.image_size + "px");
  if (m.dataset) parts.push(m.dataset);
  if (parts.length > 0) {
    var metaSpan = document.createElement("span");
    metaSpan.className = "tm-model-meta";
    metaSpan.textContent = parts.join(" · ");
    div.appendChild(metaSpan);
  }

  // Tooltip with full details
  var tip = m.name;
  if (m.image_size) tip += "\nInput: " + m.image_size + "×" + m.image_size;
  if (m.num_classes) tip += "\nClasses: " + m.num_classes;
  if (m.dataset) tip += "\nPretrained: " + m.dataset;
  tip += "\nFamily: " + m.family;
  div.title = tip;

  div.onclick = function() { tmSelectModel(m.name); };
  return div;
}

function tmSelectModel(name) {
  document.getElementById("tm-model-search").value = name;
  document.getElementById("tm-model-dropdown").classList.remove("open");
  tmGate();  // the name was set in code

  // Fetch model info for image size
  return _tmFetchJson("/api/plugins/timm/models/" + encodeURIComponent(name) + "/info")
    .then(function(info) {
      if (info.error) {
        document.getElementById("tm-model-info").textContent = "Error: " + info.error;
        return;
      }
      document.getElementById("tm-model-info").textContent =
        info.num_params_m + "M params · " + info.image_size + "px · " + info.num_features + "d embeddings";
      document.getElementById("tm-imgsz").value = info.image_size || 224;
    })
    .catch(function(e) {
      document.getElementById("tm-model-info").textContent = "Error: " + e;
    });
}

// Close dropdown on outside click
document.addEventListener("click", function(e) {
  var dd = document.getElementById("tm-model-dropdown");
  var search = document.getElementById("tm-model-search");
  if (dd && !dd.contains(e.target) && e.target !== search) {
    dd.classList.remove("open");
  }
});

// ── Presets: the form in and out ─────────────────────────────
function tmDefaultFormData() {
  return {
    run_name: "", project_name: "", model_name: "", mode: "train", task_type: "",
    train_table_url: "", val_table_url: "", use_latest: false, image_column: "", label_column: "",
    params: JSON.parse(JSON.stringify(_tmInitialParams || {})),
  };
}

// The tables the plugin was opened with: ?train=&val= from the DAG's Train link, or the action
// button's tables. They replace the preset's inputs for the page's life (SDK guide → Presets).
function tmContextInputs() {
  var ctx = (typeof PLUGIN_API !== "undefined" && PLUGIN_API.context) ? PLUGIN_API.context : {};
  var urls = ctx.resourceUrls || [];
  var q = new URLSearchParams(window.location.search);
  if (q.has("train") || q.has("val")) {
    return { train_table_url: q.get("train") || "", val_table_url: q.get("val") || "" };
  }
  if (urls.length) return { train_table_url: urls[0], val_table_url: urls[1] || "" };
  return null;
}

// The name a new preset gets unless the person types one: "resnet50 classification · 2026-10-27".
function tmAutoName(values) {
  var d = new Date();
  var dateStr = d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
  var model = (values.model_name || "timm").replace(/\..*$/, "");
  return model + " " + (values.task_type || values.mode || "train") + " \u00b7 " + dateStr;
}

// What the list shows after a preset's name: "resnet50 · classification".
function tmPresetSummary(values) {
  return [values.model_name, values.task_type || values.mode];
}

function _tmShowDetected(task, imageCol, labelCol) {
  var badge = document.getElementById("tm-task-badge");
  badge.textContent = task ? task.replace(/_/g, " ") : "--";
  badge.className = task ? "tm-task-badge-" + task : "plugin-task-badge";
  document.getElementById("tm-task-detail").textContent = "";
  document.getElementById("tm-image-col").textContent = imageCol || "--";
  document.getElementById("tm-label-col").textContent = labelCol || "--";
  document.getElementById("tm-num-classes").textContent = "";
  document.getElementById("tm-class-names").textContent = "";
}

function _tmRestoreParams(params) {
  var paramEls = document.querySelectorAll("[data-param-id]");
  for (var i = 0; i < paramEls.length; i++) {
    var el = paramEls[i];
    var val = params[el.dataset.paramId];
    if (val !== undefined) {
      if (el.type === "checkbox") el.checked = val === true || val === "true";
      else el.value = val;
    }
  }
}

// Fill the form from a preset's values. Resolves once the task is detected and the model's info
// is in, so the preset bar compares the form only then.
function tmApplyFormData(values) {
  values = values || {};
  var params = values.params || {};
  document.getElementById("tm-run-name").value = values.run_name || "";
  document.getElementById("tm-3lc-project").value = values.project_name || "";
  document.getElementById("tm-train-url").value = values.train_table_url || "";
  document.getElementById("tm-val-url").value = values.val_table_url || "";
  document.getElementById("tm-use-latest").checked = !!values.use_latest;
  document.getElementById("tm-pretrained-url").value = params.pretrained_model_url || "";
  document.getElementById("tm-copy-model").checked = params.copy_model_to_run !== false;
  tmDetectedTask = values.task_type || null;
  tmDetectedImageCol = values.image_column || "";
  tmDetectedLabelCol = values.label_column || "";
  tmTaskError = "";
  tmDetecting = !!document.getElementById("tm-train-url").value.trim();
  _tmShowDetected(values.task_type || "", tmDetectedImageCol, tmDetectedLabelCol);
  tmSetMode(values.mode || "train");
  _tmEnforceTaskMode(values.task_type || "");
  var model = values.model_name || "";
  // The train table decides the task and the columns, and the 3LC project when none is set; the
  // preset's own stand while the table cannot be read (Start then says so).
  var detect = document.getElementById("tm-train-url").value.trim() ? tmDetectTask() : Promise.resolve(false);
  return detect.then(function() {
    if (model) return tmSelectModel(model);
    document.getElementById("tm-model-search").value = "";
    document.getElementById("tm-model-info").textContent = "";
  }).then(function() {
    // The model's info sets the image size; the preset's parameters come after, so they win.
    _tmRestoreParams(params);
    tmGate();
  });
}

function tmInitPresets() {
  _tmInitialParams = tmCollectFormData().params;
  tmPresets = _tlcPluginPresets({
    pluginId: "timm",
    apiPath: "/api/plugins/timm/configs",
    mount: "tm-presets",
    formRoot: "tm-page",
    collect: tmCollectFormData,
    apply: tmApplyFormData,
    defaults: tmDefaultFormData,
    inputs: ["train_table_url", "val_table_url"],
    contextInputs: tmContextInputs,
    autoName: tmAutoName,
    summary: tmPresetSummary,
    legacyKey: _TM_STORAGE_KEY,
  });
  return tmPresets.init();
}

function tmCollectFormData() {
  var params = {};
  var paramEls = document.querySelectorAll("[data-param-id]");
  for (var i = 0; i < paramEls.length; i++) {
    var el = paramEls[i];
    params[el.dataset.paramId] = el.type === "checkbox" ? el.checked : el.value;
  }

  // Model fine-tuning and checkpoint params
  var pretrainedUrl = document.getElementById("tm-pretrained-url").value.trim();
  if (pretrainedUrl) params.pretrained_model_url = pretrainedUrl;
  params.copy_model_to_run = document.getElementById("tm-copy-model").checked;

  return {
    run_name: document.getElementById("tm-run-name").value.trim(),
    project_name: document.getElementById("tm-3lc-project").value.trim(),
    model_name: document.getElementById("tm-model-search").value.trim(),
    mode: tmCurrentMode,
    task_type: tmDetectedTask || "",
    train_table_url: document.getElementById("tm-train-url").value.trim(),
    val_table_url: document.getElementById("tm-val-url").value.trim(),
    use_latest: document.getElementById("tm-use-latest").checked,
    image_column: tmDetectedImageCol,
    label_column: tmDetectedLabelCol,
    params: params,
  };
}

// ── Task Detection ───────────────────────────────────────────
var _tmDetectDebounce = null;

function tmOnTrainTableChange() {
  // Another table: its task and columns are unknown until detection answers, and the previous
  // table's must not ride along in a run. Start waits for the answer, debounce included.
  var url = document.getElementById("tm-train-url").value.trim();
  tmDetectedTask = null;
  tmDetectedImageCol = "";
  tmDetectedLabelCol = "";
  tmTaskError = "";
  tmDetecting = !!url;
  _tmShowDetected("", "", "");
  if (url) document.getElementById("tm-task-badge").textContent = "detecting...";
  tmGate();
  clearTimeout(_tmDetectDebounce);
  _tmDetectDebounce = setTimeout(tmDetectTask, 500);
}

// Detect the task and the columns from the train table; resolves true when it answered. Fills the
// 3LC project when none is set. A table that cannot be read leaves the task and columns as they
// are and sets tmTaskError.
function tmDetectTask() {
    var url = document.getElementById("tm-train-url").value.trim();
    tmTaskError = "";
    if (!url) {
      tmDetectedTask = null;
      tmDetectedImageCol = "";
      tmDetectedLabelCol = "";
      tmDetecting = false;
      _tmShowDetected("", "", "");
      tmGate();
      return Promise.resolve(false);
    }
    tmDetecting = true;
    document.getElementById("tm-task-badge").textContent = "detecting...";
    tmGate();
    // An answer for a table no longer in the field is dropped: that field's own detection answers.
    var current = function() { return document.getElementById("tm-train-url").value.trim() === url; };
    var failed = function(error) {
      tmDetecting = false;
      tmTaskError = error;
      _tmSetTaskBadge("error", error);
      tmGate();
      return false;
    };

    return _tmPost("/api/plugins/timm/detect-task", { url: url })
      .then(function(data) {
        if (!current()) return false;
        if (!data || data.error || !data.task) {
          return failed(String((data && (data.error || data.detail)) || "No task in the answer."));
        }
        tmDetecting = false;
        tmDetectedTask = data.task;
        tmDetectedImageCol = data.image_column || "";
        tmDetectedLabelCol = data.label_column || "";

        _tmSetTaskBadge(data.task, data.num_classes ? data.num_classes + " classes" : "");
        document.getElementById("tm-image-col").textContent = data.image_column || "--";
        document.getElementById("tm-label-col").textContent = data.label_column || "--";
        document.getElementById("tm-num-classes").textContent = data.num_classes ? " (" + data.num_classes + " classes)" : "";

        // Show class names
        var classEl = document.getElementById("tm-class-names");
        if (data.class_names && data.class_names.length > 0 && data.class_names.length <= 30) {
          classEl.textContent = data.class_names.join(", ");
        } else if (data.class_names && data.class_names.length > 30) {
          classEl.textContent = data.class_names.slice(0, 30).join(", ") + " … +" + (data.class_names.length - 30) + " more";
        } else {
          classEl.textContent = "";
        }

        // Auto-set mode based on task: feature_extraction → collect only
        _tmEnforceTaskMode(data.task);

        if (data.project_name) {
          var projEl = document.getElementById("tm-3lc-project");
          if (!projEl.value.trim()) projEl.value = data.project_name;
        }
        tmGate();
        return true;
      })
      .catch(function(e) {
        return current() ? failed(String(e)) : false;
      });
}

// ── Training ─────────────────────────────────────────────────
function tmStartTraining() {
  // Start is disabled while it cannot run; a stale state must not submit either.
  if (_tmSubmitting || tmStartBlocker()) { tmGate(); return; }
  var data = tmCollectFormData();
  // Launch through the SDK's PluginJobs client. project_name attributes the job in the generic
  // Queue & Progress panel — the host reads it from the run body only. Left blank, PluginJobs
  // fills it from the launch context (the project page the plugin was opened from). /run is
  // answered by the host itself (the job is queued, the worker is spawned later), so authFetch's
  // default timeout is enough here — the generous _tmFetch timeout is for the worker-proxied
  // routes only.
  // The run body is self-contained: project_config rides inline, so a remote GPU worker needs no
  // local config store (SDK guide, "Run-body conventions for remote workers"). The preset's id
  // and name are set while the form is unchanged from it, so the job card can name it, and empty
  // otherwise. A run never saves a preset.
  var presetId = tmPresets ? tmPresets.presetId() : "";
  var config = Object.assign({}, data, { id: presetId, name: tmPresets ? tmPresets.presetName() : "" });
  // One run per press: Start stays disabled until the run request answers.
  _tmSubmitting = true;
  tmGate();
  PluginJobs.start("timm", {
    config_id: presetId,
    project_name: (document.getElementById("tm-3lc-project").value || "").trim(),
    project_config: config,
  })
    .then(function(trainResp) {
      _tmSubmitting = false;
      tmGate();
      if (!trainResp || trainResp.error) {
        tmLog("Error: " + ((trainResp && trainResp.error) || "failed to start job"));
        return;
      }
      if (!trainResp.job_id) { tmLog(_tmNotStartedMessage(trainResp)); return; }
      tmLog("timm job submitted: " + trainResp.job_id);
      tmShowCurrentJob(trainResp.job_id, _tmJobLabel(config));
      _tmTrackJob(trainResp.job_id);
    })
    .catch(function(e) { _tmSubmitting = false; tmGate(); tmLog("Error starting: " + e); });
}

function tmStopCurrentJob() {
  if (!_tmCurrentJobId) { console.warn("tmStopCurrentJob: no _tmCurrentJobId"); return; }
  CancelJob.show(_tmCurrentJobId, {
    pluginId: 'timm',
    onDone: function(result) {
      tmLog(result.deleted ? "Job stopped and run deleted." : "Job stopped.");
      tmLoadQueue();
    }
  });
}

function tmSwapUrls() {
  var t = document.getElementById("tm-train-url");
  var v = document.getElementById("tm-val-url");
  var tmp = t.value; t.value = v.value; v.value = tmp;
  tmOnTrainTableChange();
}

// /run answers {status, provisioning} instead of a job while the plugin's
// environment is still being built (or its last build failed) — say so, rather
// than tracking an undefined job id.
function _tmNotStartedMessage(resp) {
  var prov = resp.provisioning || {};
  var state = prov.state || resp.status || "unknown";
  if (state === "failed") {
    return "Plugin environment build failed" + (prov.error ? ": " + prov.error : "") +
      " — see the Plugins page.";
  }
  if (state === "provisioning" || state === "unprovisioned") {
    return "Plugin environment is still being set up (" + state + ") — try again in a moment.";
  }
  return "Job was not started (status: " + state + ").";
}

// ── Queue Rendering ──────────────────────────────────────────
function tmLoadQueue() {
  // The per-plugin /queue endpoint was removed in the run_job migration; the
  // generic Queue & Progress panel (host) now owns queue + cancel. Live training
  // updates still arrive via the epoch_progress / job_status socket handlers.
  // No-op kept so the existing call sites stay valid.
}

// Build a human label for the current-job panel from the submitted form data.
function _tmJobLabel(data) {
  var modeLabel = data.mode === "collect" ? "Collect" : "Train";
  var name = (data.name || data.run_name || "Job").trim();
  return name + " (" + (data.model_name || "timm") + " · " + modeLabel + ")";
}

// Reveal the current-job panel. The backend no longer emits the legacy
// queue_update event (removed in the run_job migration), so the live
// epoch_progress / job_status socket handlers drive visibility instead — every
// one of those events carries data.job_id.
function tmShowCurrentJob(jobId, label) {
  if (jobId) { _tmCurrentJobId = jobId; }
  var currentDiv = document.getElementById("tm-current-job");
  var noCurrentDiv = document.getElementById("tm-no-current");
  var firstReveal = currentDiv.style.display === "none" || currentDiv.style.display === "";
  if (firstReveal) {
    currentDiv.style.display = "block";
    noCurrentDiv.style.display = "none";
    var statusEl = document.getElementById("tm-current-status");
    statusEl.textContent = "running";
    statusEl.className = "tm-queue-status plugin-status-running";
    document.getElementById("tm-stop-btn").style.display = "inline-block";
  }
  if (label) { document.getElementById("tm-current-label").textContent = label; }
}

// Mark the current-job panel terminal (completed / failed / cancelled) from the
// generic job record: `run_url` drives the Open link, `error` the failure text.
function tmFinishCurrentJob(status, job) {
  job = job || {};
  var statusEl = document.getElementById("tm-current-status");
  statusEl.textContent = status;
  statusEl.className = "tm-queue-status plugin-status-" + status;
  document.getElementById("tm-stop-btn").style.display = "none";
  var runLinkEl = document.getElementById("tm-current-run-link");
  if (runLinkEl) {
    var projectName = (document.getElementById("tm-3lc-project").value || "").trim();
    var href = _tmOpenHref(job, projectName);
    if (href) {
      runLinkEl.href = href;
      runLinkEl.style.display = "inline";
    } else {
      runLinkEl.style.display = "none";
    }
  }
  var infoEl = document.getElementById("tm-current-info");
  if (infoEl) {
    var errText = status === "failed" ? (job.error || job.subtitle || "") : "";
    infoEl.textContent = errText;
    infoEl.style.color = errText ? "var(--danger-color)" : "";
  }
  _tmRenderLiveBanner(job.status ? job : { status: status });
  _tmCurrentJobId = null;
}

function tmRenderQueue(data) {
  var currentDiv = document.getElementById("tm-current-job");
  var noCurrentDiv = document.getElementById("tm-no-current");
  var stopBtn = document.getElementById("tm-stop-btn");

  if (data.current) {
    currentDiv.style.display = "block";
    noCurrentDiv.style.display = "none";
    _tmCurrentJobId = data.current.id;
    var modeLabel = data.current.mode === "collect" ? "Collect" : "Train";
    document.getElementById("tm-current-label").textContent =
      (data.current.tlc_run_name || data.current.config_name || "Job") + " (" + data.current.model_name + " · " + modeLabel + ")";
    var statusEl = document.getElementById("tm-current-status");
    statusEl.textContent = data.current.status;
    statusEl.className = "tm-queue-status plugin-status-" + data.current.status;
    stopBtn.style.display = data.current.status === "running" ? "inline-block" : "none";
    // Populate progress bar / epoch info from HTTP data so it's correct on page load
    // (SocketIO epoch_progress will keep it live afterwards)
    var p = data.current.progress || {};
    if (p.epoch && p.total_epochs) {
      var m = p.metrics || {};
      var batch = m.batch || 0;
      var totalBatches = m.total_batches || 0;
      var phase = m.phase || "";
      var epochFraction = totalBatches > 0 ? (batch / totalBatches) : 1;
      var overallPct = Math.min(((p.epoch - 1 + epochFraction) / p.total_epochs) * 100, 100);
      document.getElementById("tm-epoch-bar").style.width = overallPct.toFixed(1) + "%";
      document.getElementById("tm-pct-label").textContent = Math.round(overallPct) + "%";
      if (batch && totalBatches && phase) {
        var phaseLabel = phase === "val" ? "Val" : "Train";
        document.getElementById("tm-epoch-label").textContent =
          "Epoch " + p.epoch + "/" + p.total_epochs + " · " + phaseLabel + " batch " + batch + "/" + totalBatches;
      } else {
        document.getElementById("tm-epoch-label").textContent = "Epoch " + p.epoch + "/" + p.total_epochs;
      }
      var timing = p.timing || {};
      var parts = [];
      if (timing.elapsed_s !== undefined) parts.push("Elapsed: " + _tmFmtDuration(timing.elapsed_s));
      if (timing.eta_s !== undefined) parts.push("ETA: " + _tmFmtDuration(timing.eta_s));
      if (timing.avg_epoch_s !== undefined) parts.push("Per epoch: " + _tmFmtDuration(timing.avg_epoch_s));
      document.getElementById("tm-timing").innerHTML = parts.join(' &nbsp;|&nbsp; ');
      tmRenderMetrics(m);
    } else if (p.collecting_status) {
      document.getElementById("tm-epoch-label").textContent = "Collecting metrics";
      document.getElementById("tm-epoch-bar").style.width = "100%";
      document.getElementById("tm-pct-label").textContent = "";
    }
    // Show "View Run" link if run_url is available (even while running)
    var runLinkEl = document.getElementById("tm-current-run-link");
    if (runLinkEl) {
      if (data.current.tlc_project_name) {
        runLinkEl.href = "/projects/" + encodeURIComponent(data.current.tlc_project_name);
        runLinkEl.style.display = "inline";
      } else {
        runLinkEl.style.display = "none";
      }
    }
  } else {
    currentDiv.style.display = "none";
    noCurrentDiv.style.display = "block";
    _tmCurrentJobId = null;
    stopBtn.style.display = "none";
  }

  // Queued + completed rendering follows same pattern as YOLO
  var queueSection = document.getElementById("tm-queue-section");
  var queueList = document.getElementById("tm-queue-list");
  if (data.queued && data.queued.length > 0) {
    queueSection.style.display = "block";
    queueList.innerHTML = "";
    data.queued.forEach(function(j) {
      var item = document.createElement("div");
      item.className = "tm-queue-item";
      item.innerHTML = '<span class="tm-queue-status plugin-status-queued">queued</span>' +
        '<span style="flex:1">' + _esc(j.tlc_run_name || j.config_name) + '</span>' +
        '<button class="btn btn-ghost btn-sm" onclick="tmCancelJob(\'' + j.id + '\')" style="color:var(--danger-color)">Cancel</button>';
      queueList.appendChild(item);
    });
  } else {
    queueSection.style.display = "none";
  }

  var completedSection = document.getElementById("tm-completed-section");
  var completedList = document.getElementById("tm-completed-list");
  if (data.completed && data.completed.length > 0) {
    completedSection.style.display = "block";
    completedList.innerHTML = "";
    data.completed.slice().reverse().forEach(function(j) {
      var item = document.createElement("div");
      item.className = "tm-queue-item";
      var statusClass = j.status === "completed" ? "plugin-status-completed" :
                        j.status === "failed" ? "plugin-status-failed" : "plugin-status-" + j.status;
      var runLink = "";
      if (j.tlc_project_name) {
        runLink = ' <a class="btn btn-sm btn-secondary" href="/projects/' + encodeURIComponent(j.tlc_project_name) +
                  '">Open</a>';
      }
      var errorMsg = j.error ? '<div style="font-size:10px;color:var(--danger-color);margin-top:2px">' + _esc(j.error) + '</div>' : "";
      // Show key metrics for completed jobs
      var metricsHtml = "";
      var pm = (j.progress || {}).metrics || {};
      if (j.status === "completed" && pm.val_acc !== undefined) {
        metricsHtml = '<span class="text-sm text-muted" style="white-space:nowrap">' +
          'val_acc ' + Number(pm.val_acc).toFixed(1) + '%' +
          (pm.val_loss !== undefined ? ' · loss ' + Number(pm.val_loss).toFixed(3) : '') + '</span>';
      }
      item.innerHTML =
        '<span class="tm-queue-status ' + statusClass + '">' + j.status + '</span>' +
        '<span style="flex:1;min-width:0"><span style="font-size:11px">' + _esc(j.tlc_run_name || j.config_name) +
        '</span> <span class="text-muted" style="font-size:10px">(' + _esc(j.model_name) + ')</span>' +
        runLink + errorMsg + '</span>' + metricsHtml;
      completedList.appendChild(item);
    });
  } else {
    completedSection.style.display = "none";
  }
}

function tmCancelJob(jobId) {
  _tmPost("/api/plugins/jobs/" + jobId + "/cancel", {})
    .then(function(r) { if (r.cancelled) tmLog("Job cancelled."); })
    .catch(function(e) { tmLog("Cancel error: " + e); });
}

// ── Metrics Rendering ────────────────────────────────────────
function tmRenderMetrics(m) {
  var container = document.getElementById("tm-current-metrics");
  if (!container) return;
  var skip = { phase: 1, batch: 1, total_batches: 1, batch_frac: 1 };
  // Metrics that are percentages (append %)
  var pctKeys = { val_acc: 1, train_acc: 1 };
  // Preferred display order
  var order = ["train_loss", "val_loss", "train_acc", "val_acc", "lr"];
  var keys = order.filter(function(k) { return m[k] !== undefined; });
  // Add any remaining keys not in the preferred order
  Object.keys(m).forEach(function(k) {
    if (!skip[k] && typeof m[k] === "number" && keys.indexOf(k) === -1) keys.push(k);
  });

  var html = "";
  keys.forEach(function(k) {
    var v = m[k];
    var val;
    if (pctKeys[k]) {
      val = v.toFixed(1) + "%";
    } else if (k === "lr") {
      val = v.toExponential(1);
    } else if (v < 0.01 && v > 0) {
      val = v.toExponential(2);
    } else {
      val = v < 1 ? v.toFixed(4) : v.toFixed(2);
    }
    html += '<div class="plugin-metric-card"><div class="plugin-metric-value">' + val + '</div>' +
            '<div class="plugin-metric-label">' + _esc(k.replace(/_/g, " ")) + '</div></div>';
  });
  container.innerHTML = html;
}

// ── Auto-save ───────────────────────────────────────────────
// The SDK's preset bar owns the saved setups (SDK guide → Presets). The form's handlers call this
// after a change, so the bar can compare the form; a run never saves a preset.
function tmScheduleAutoSave() {
  if (tmPresets) tmPresets.changed();
}

function _tmBindAutoSave() {
  // Bind auto-save to all form inputs
  var inputs = document.querySelectorAll("#tm-run-name, #tm-3lc-project, #tm-model-search, [data-param-id]");
  for (var i = 0; i < inputs.length; i++) {
    inputs[i].addEventListener("input", tmScheduleAutoSave);
    inputs[i].addEventListener("change", tmScheduleAutoSave);
  }
  document.getElementById("tm-use-latest").addEventListener("change", tmScheduleAutoSave);
}


// ── Data Source Init ──────────────────────────────────────────
// The SDK's shared data-source picker for the pretrained checkpoint: the rendered
// text input keeps the id "tm-pretrained-url", so config save/restore reads and
// writes it exactly as before.
(function() {
  document.getElementById("tm-pretrained-ds").innerHTML = _tlcDataSourceHtml("tm-pretrained-url", {
    placeholder: "Path to .pt checkpoint (leave empty for ImageNet pretrained)",
    accept: "*.pt,*.pth",
    mode: "file",
  });
  _tlcBindDataSource("tm-pretrained-url", TM_COMPUTE_URL, "timm", { accept: "*.pt,*.pth", mode: "file" });
})();

// ── Init ─────────────────────────────────────────────────────
(function() {
  // Start and Swap follow every edit; what the page sets in code calls tmGate itself.
  document.getElementById("tm-page").addEventListener("input", tmGate);
  document.getElementById("tm-page").addEventListener("change", tmGate);
  tmGate();
  tmInitSocket();
  _tmSeedFromHost();
  tmLoadModels();
  tmLoadQueue();
  _tmBindAutoSave();

  // The preset bar opens the form: as it was last left in this browser, else on Default
  // settings, with the tables the plugin was opened with (tmContextInputs) in place of its own.
  tmInitPresets();
})();
