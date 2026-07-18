let anomalyPayload = null;
let productPayload = null;
let productChart = null;
let systemicChart = null;
let autoencoderChart = null;
let contextChart = null;
let confirmationChart = null;

async function loadAnomalyStatus() {
  const response = await fetch("/api/anomaly-lab/status", { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function loadAnomalyLab() {
  const response = await fetch("/api/anomaly-lab", { cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function n(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function yesNo(value) {
  return value ? "Yes" : "No";
}

function compactNumber(value, digits = 1) {
  const number = n(value);
  if (number === null) return "—";
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function ratioPct(value, digits = 1) {
  const number = n(value);
  if (number === null) return "—";
  return `${(number * 100).toFixed(digits)}%`;
}

function signedPct(value, digits = 1) {
  const number = n(value);
  if (number === null) return "—";
  const pctValue = number * 100;
  return `${pctValue >= 0 ? "+" : ""}${pctValue.toFixed(digits)}%`;
}

function wape(value) {
  return ratioPct(value, 2);
}

function statusClass(kind) {
  return kind === "good" ? "status-good" : kind === "bad" ? "status-bad" : "status-running";
}

function renderDecision(data) {
  const card = document.getElementById("anomaly-decision-card");
  const dot = card.querySelector(".status-dot");
  const title = document.getElementById("anomaly-decision-title");
  const copy = document.getElementById("anomaly-decision-copy");
  const weekend = data.weekend_v2;
  const overnight = data.overnight;

  let state = "running";
  if (weekend?.state === "complete" && weekend.recommendation) {
    const promote = weekend.recommendation.promote_weekend_v2 === true;
    state = promote ? "good" : "bad";
    title.textContent = promote
      ? `Weekend-v2 promoted ${weekend.recommendation.winner?.name || "a specialist policy"}`
      : "Weekend-v2 retained the canonical control";
    copy.textContent = promote
      ? "The anomaly/regime specialist policy passed development, benchmark, high-volume, holiday and bootstrap gates."
      : "No specialist policy passed every safety gate; anomaly signals remain diagnostic rather than part of the submitted point forecast.";
  } else if (weekend?.state === "running") {
    title.textContent = "Weekend-v2 search is running";
    copy.textContent = "The page will update automatically when screening, refinement, confirmation and ensemble selection write new artifacts.";
  } else if (overnight?.recommendation) {
    const winner = overnight.recommendation.winner?.name || "control";
    state = "bad";
    title.textContent = `Overnight winner: ${winner}`;
    copy.textContent = "Standalone anomaly candidates did not pass confirmation, but complementary residual structure justified the specialist-focused weekend-v2 search.";
  } else {
    title.textContent = "Anomaly outputs are not available yet";
    copy.textContent = "Run the anomaly audit and search pipelines to populate this page.";
  }
  dot.className = `status-dot ${statusClass(state)}`;
}

function renderKpis(data) {
  const audit = data.audit || {};
  const metadata = audit.metadata || {};
  const localEvt = metadata.local_evt || {};
  const systemicEvt = metadata.systemic_evt || {};
  const context = metadata.test_context || {};
  const aeRuns = data.autoencoder?.runs || [];
  const primaryAe = aeRuns[0]?.metadata || {};
  const weekendState = data.weekend_v2?.state || "not started";
  const cards = [
    {
      label: "Local anomalies",
      value: compactNumber(metadata.n_local_anomalies, 0),
      sub: `${ratioPct((n(metadata.n_local_anomalies, 0) || 0) / Math.max(n(metadata.n_scored, 1) || 1, 1), 2)} of scored product-days`,
    },
    {
      label: "Systemic days",
      value: compactNumber(metadata.n_systemic_days, 0),
      sub: `Validation FAR ${ratioPct(systemicEvt.validation_far, 2)}`,
    },
    {
      label: "Event-protected flags",
      value: compactNumber(audit.event_protected_anomalies, 0),
      sub: "Preserved as possible commercial signal",
    },
    {
      label: "Test context shifts",
      value: compactNumber(context.n_shift_flags, 0),
      sub: `Max novelty percentile ${ratioPct(context.max_percentile, 1)}`,
    },
    {
      label: "AE windows flagged",
      value: compactNumber(primaryAe.n_flagged, 0),
      sub: `${compactNumber(primaryAe.n_windows, 0)} temporal windows scored`,
    },
    {
      label: "Weekend-v2",
      value: String(weekendState).replaceAll("_", " "),
      sub: `Local threshold ${compactNumber(localEvt.threshold, 2)}`,
    },
  ];
  document.getElementById("anomaly-kpi-grid").innerHTML = cards.map((card) => `
    <div class="kpi-card anomaly-kpi-card">
      <p class="kpi-label">${esc(card.label)}</p>
      <p class="kpi-value">${esc(card.value)}</p>
      <p class="kpi-sub">${esc(card.sub)}</p>
    </div>
  `).join("");
}

function renderProductOptions(data) {
  const select = document.getElementById("anomaly-product-select");
  const rows = data.audit?.product_summary || [];
  const products = data.audit?.products || rows.map((row) => row.ProductId);
  select.innerHTML = products.map((productId) => {
    const row = rows.find((item) => Number(item.ProductId) === Number(productId));
    const count = row ? ` · ${row.local_anomalies || 0} flags` : "";
    return `<option value="${productId}">Product ${productId}${count}</option>`;
  }).join("");
  if (products.length) select.value = String(products[0]);
}

async function loadProduct(productId) {
  const response = await fetch(`/api/anomaly-lab/product/${encodeURIComponent(productId)}`, { cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Could not load product ${productId}`);
  }
  productPayload = await response.json();
  renderProduct();
}

function productWindowRows(rows) {
  const range = document.getElementById("anomaly-range-select").value;
  if (range === "all") return rows;
  return rows.slice(-Number(range));
}

function renderProduct() {
  if (!productPayload?.available) return;
  const summary = productPayload.summary || {};
  document.getElementById("anomaly-product-summary").innerHTML = [
    ["Observed days", compactNumber(summary.observed_days, 0)],
    ["Local flags", compactNumber(summary.local_anomalies, 0)],
    ["Event-associated", compactNumber(summary.event_anomalies, 0)],
    ["Maximum score", compactNumber(summary.max_score, 2)],
    ["Mean training weight", compactNumber(summary.mean_weight, 3)],
  ].map(([label, value]) => `<span><small>${esc(label)}</small><strong>${esc(value)}</strong></span>`).join("");

  const rows = productWindowRows(productPayload.timeline || []);
  const labels = rows.map((row) => row.DateKey);
  const actual = rows.map((row) => n(row.Quantity));
  const expected = rows.map((row) => n(row.expected_quantity));
  const score = rows.map((row) => n(row.anomaly_score));
  const flags = rows.map((row) => row.anomaly_flag ? n(row.Quantity) : null);

  if (productChart) productChart.destroy();
  productChart = new Chart(document.getElementById("chart-product-anomalies"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Observed quantity", data: actual, borderColor: "#0a0a0a", backgroundColor: "#0a0a0a", pointRadius: 0, borderWidth: 1.8, yAxisID: "y" },
        { label: "Causal expectation", data: expected, borderColor: "#64748b", backgroundColor: "#64748b", pointRadius: 0, borderDash: [5, 4], borderWidth: 1.4, yAxisID: "y" },
        { label: "Anomaly score", data: score, borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,0.12)", pointRadius: 0, borderWidth: 1.2, yAxisID: "y1" },
        { label: "Flagged observation", data: flags, borderColor: "#c2410c", backgroundColor: "#c2410c", showLine: false, pointRadius: 4, pointHoverRadius: 6, yAxisID: "y" },
      ],
    },
    options: anomalyLineOptions("Quantity", "Anomaly score"),
  });

  const tbody = document.querySelector("#product-anomaly-table tbody");
  tbody.innerHTML = (productPayload.top_anomalies || []).map((row) => `
    <tr>
      <td>${esc(row.DateKey)}</td>
      <td>${compactNumber(row.Quantity, 1)}</td>
      <td>${compactNumber(row.expected_quantity, 1)}</td>
      <td><strong>${compactNumber(row.anomaly_score, 2)}</strong></td>
      <td>${compactNumber(row.anomaly_signed_residual, 2)}</td>
      <td>${row.known_event ? '<span class="tag tag-event">Known event</span>' : '<span class="tag">Unexplained</span>'}</td>
      <td>${compactNumber(row.anomaly_weight, 3)}</td>
    </tr>
  `).join("") || '<tr><td colspan="7">No local anomalies for this product.</td></tr>';
}

function anomalyLineOptions(leftTitle, rightTitle) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    animation: false,
    scales: {
      x: { ticks: { maxTicksLimit: 12 }, grid: { color: "#eeeeee" } },
      y: { position: "left", title: { display: true, text: leftTitle }, beginAtZero: true, grid: { color: "#e8e8e8" } },
      y1: { position: "right", title: { display: true, text: rightTitle }, beginAtZero: true, grid: { drawOnChartArea: false } },
    },
    plugins: { legend: { position: "bottom" } },
  };
}

function renderSystemic(data) {
  const rows = data.audit?.daily || [];
  if (!rows.length) return;
  if (systemicChart) systemicChart.destroy();
  systemicChart = new Chart(document.getElementById("chart-systemic-timeline"), {
    type: "line",
    data: {
      labels: rows.map((row) => row.DateKey),
      datasets: [
        { label: "Total quantity", data: rows.map((row) => n(row.total_quantity)), borderColor: "#0a0a0a", backgroundColor: "#0a0a0a", pointRadius: 0, borderWidth: 1.5, yAxisID: "y" },
        { label: "Local anomaly count", data: rows.map((row) => n(row.local_anomalies)), borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,0.12)", pointRadius: 0, borderWidth: 1.2, yAxisID: "y1" },
        { label: "Systemic flag", data: rows.map((row) => row.systemic_flag ? n(row.total_quantity) : null), borderColor: "#dc2626", backgroundColor: "#dc2626", showLine: false, pointRadius: 3, yAxisID: "y" },
      ],
    },
    options: anomalyLineOptions("Daily quantity", "Flagged products"),
  });

  document.querySelector("#top-local-table tbody").innerHTML = (data.audit?.top_local || []).map((row) => {
    const interpretation = row.known_event
      ? '<span class="tag tag-event">Explainable event</span>'
      : row.systemic_anomaly_flag
        ? '<span class="tag tag-systemic">Systemic</span>'
        : '<span class="tag">Local only</span>';
    return `<tr>
      <td>${esc(row.ProductId)}</td><td>${esc(row.DateKey)}</td>
      <td>${compactNumber(row.Quantity, 1)}</td><td>${compactNumber(row.expected_quantity, 1)}</td>
      <td><strong>${compactNumber(row.anomaly_score, 2)}</strong></td><td>${interpretation}</td>
    </tr>`;
  }).join("");
}

function autoencoderLabel(run) {
  const cfg = run.metadata?.config || {};
  const trainPct = n(cfg.train_fraction) !== null ? `${Math.round(cfg.train_fraction * 100)}% train` : run.name;
  return `${run.name.replace("anomaly_autoencoder_", "")} · ${trainPct}`;
}

function splitRows(run) {
  const interpretation = run.interpretation || {};
  return interpretation.ae_split_summary || interpretation.split_summary || [];
}

function renderAutoencoderSelector(data) {
  const select = document.getElementById("autoencoder-run-select");
  const runs = data.autoencoder?.runs || [];
  select.innerHTML = runs.map((run, index) => `<option value="${index}">${esc(autoencoderLabel(run))}</option>`).join("");
  select.disabled = runs.length < 2;
  renderAutoencoderRun(0);
}

function renderAutoencoderRun(index) {
  const run = anomalyPayload?.autoencoder?.runs?.[index];
  if (!run) return;
  const metadata = run.metadata || {};
  const evt = metadata.evt || {};
  const cfg = metadata.config || {};
  const interpretation = run.interpretation || {};
  const drift = n(interpretation.score_date_correlation);
  const far = n(evt.validation_far);
  const targetFar = n(evt.alpha);
  const strip = [
    ["Architecture", cfg.architecture || "MLP"],
    ["Windows", compactNumber(metadata.n_windows, 0)],
    ["Epochs", compactNumber(cfg.epochs || cfg.max_epochs, 0)],
    ["Threshold", compactNumber(evt.threshold, 3)],
    ["Validation FAR", ratioPct(far, 1)],
    ["Target FAR", ratioPct(targetFar, 1)],
    ["Time drift", compactNumber(drift, 3)],
    ["Flagged", compactNumber(metadata.n_flagged, 0)],
  ];
  document.getElementById("autoencoder-diagnostic-strip").innerHTML = strip.map(([label, value]) => `
    <span><small>${esc(label)}</small><strong>${esc(value)}</strong></span>
  `).join("");

  const rows = run.timeline || [];
  const threshold = n(evt.threshold);
  if (autoencoderChart) autoencoderChart.destroy();
  autoencoderChart = new Chart(document.getElementById("chart-autoencoder"), {
    type: "line",
    data: {
      labels: rows.map((row) => row.DateKey),
      datasets: [
        { label: "Reconstruction score", data: rows.map((row) => n(row.systemic_autoencoder_score)), borderColor: "#7c3aed", backgroundColor: "rgba(124,58,237,0.10)", pointRadius: 0, borderWidth: 1.4, yAxisID: "y" },
        { label: "Calibrated threshold", data: rows.map(() => threshold), borderColor: "#dc2626", pointRadius: 0, borderWidth: 1, borderDash: [6, 4], yAxisID: "y" },
        { label: "Total quantity", data: rows.map((row) => n(row.total_quantity)), borderColor: "#9ca3af", pointRadius: 0, borderWidth: 1, yAxisID: "y1" },
        { label: "Flagged window", data: rows.map((row) => row.systemic_autoencoder_flag ? n(row.systemic_autoencoder_score) : null), borderColor: "#f59e0b", backgroundColor: "#f59e0b", showLine: false, pointRadius: 3, yAxisID: "y" },
      ],
    },
    options: anomalyLineOptions("Reconstruction error", "Daily quantity"),
  });

  const split = splitRows(run);
  document.querySelector("#autoencoder-split-table tbody").innerHTML = split.map((row) => `
    <tr><td>${esc(row.autoencoder_split || row.split)}</td><td>${compactNumber(row.n, 0)}</td>
      <td>${compactNumber(row.flagged, 0)}</td><td>${ratioPct(row.flag_rate, 1)}</td>
      <td>${compactNumber(row.mean_score, 3)}</td><td>${compactNumber(row.max_score, 3)}</td></tr>
  `).join("");

  const unstableFar = far !== null && targetFar !== null && far > targetFar * 2;
  const driftWarning = drift !== null && Math.abs(drift) > 0.35;
  let headline = "The score behaves as a diagnostic regime indicator.";
  let details = "Its binary flags should be interpreted only after temporal calibration and split-stability checks.";
  if (unstableFar) {
    headline = "Calibration warning: the observed false-alarm rate is much higher than targeted.";
    details = "This run detects genuine distribution change, but its threshold is not trustworthy enough for automatic actions.";
  } else if (metadata.n_flagged === 0 || metadata.n_flagged === 1) {
    headline = "Recent training largely absorbs the apparent anomalies.";
    details = "The sharp reduction in flags under a later split suggests persistent temporal drift rather than stable isolated anomalies.";
  }
  document.getElementById("autoencoder-interpretation").innerHTML = `
    <strong>${esc(headline)}</strong><span>${esc(details)}</span>
    ${driftWarning ? '<span class="warning-line">The score is materially correlated with time, so regime drift is a competing explanation.</span>' : ''}
  `;
}

function renderContext(data) {
  const rows = data.audit?.context_daily || [];
  if (contextChart) contextChart.destroy();
  contextChart = new Chart(document.getElementById("chart-context-risk"), {
    type: "line",
    data: {
      labels: rows.map((row) => row.DateKey),
      datasets: [
        { label: "Mean novelty percentile", data: rows.map((row) => n(row.mean_context_risk)), borderColor: "#0a0a0a", backgroundColor: "#0a0a0a", pointRadius: 3, borderWidth: 1.8 },
        { label: "Maximum novelty percentile", data: rows.map((row) => n(row.max_context_risk)), borderColor: "#f59e0b", backgroundColor: "#f59e0b", pointRadius: 3, borderWidth: 1.8 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { min: 0, max: 1, ticks: { callback: (value) => `${Math.round(value * 100)}%` } } },
      plugins: { legend: { position: "bottom" } },
    },
  });
  const context = data.audit?.metadata?.test_context || {};
  document.getElementById("context-risk-note").innerHTML = `
    <strong>${compactNumber(context.n_shift_flags, 0)} of ${compactNumber(context.n_rows, 0)} rows exceeded the shift threshold.</strong>
    <span>The supplied test week is therefore not globally out of distribution on known commercial context, although individual rows can still be unusual.</span>
  `;
  document.querySelector("#context-risk-table tbody").innerHTML = (data.audit?.context_rows || []).slice(0, 10).map((row) => `
    <tr><td>${esc(row.ProductId)}</td><td>${esc(row.DateKey)}</td><td>${ratioPct(row.context_risk_percentile, 1)}</td>
      <td>${compactNumber(row.context_risk_raw, 3)}</td><td>${row.context_shift_flag ? '<span class="tag tag-bad">Shift</span>' : 'No'}</td></tr>
  `).join("");
}

function renderOvernight(data) {
  const overnight = data.overnight || {};
  const counts = overnight.counts || {};
  document.getElementById("overnight-funnel").innerHTML = [
    ["Diagnostic", counts.diagnostic, "Representations"],
    ["Proxy", counts.proxy, "Forecast policies"],
    ["Neural", counts.neural, "MPS NeuralNets"],
    ["Confirmation", counts.confirmation, "Multi-seed finalists"],
  ].map(([label, value, sub], index) => `
    <div class="funnel-stage"><span>${index + 1}</span><strong>${compactNumber(value, 0)} ${esc(label)}</strong><small>${esc(sub)}</small></div>
  `).join("");
  const promote = overnight.recommendation?.promote_anomaly_layer === true;
  const badge = document.getElementById("overnight-status-badge");
  badge.textContent = promote ? "Anomaly layer promoted" : "Control retained";
  badge.classList.toggle("winner-badge-good", promote);

  document.querySelector("#diagnostic-leaderboard-table tbody").innerHTML = (overnight.diagnostic_leaderboard || []).map((row) => `
    <tr><td>${esc(row.name)}</td><td>${compactNumber(row.future_error_top_decile_lift, 2)}×</td>
      <td>${compactNumber(row.future_wape_spearman, 3)}</td><td>${compactNumber(row.seed_stability, 3)}</td>
      <td>${compactNumber(row.time_drift_spearman, 3)}</td></tr>
  `).join("");

  const rows = overnight.confirmation_leaderboard || [];
  if (confirmationChart) confirmationChart.destroy();
  confirmationChart = new Chart(document.getElementById("chart-confirmation"), {
    type: "bar",
    data: {
      labels: rows.map((row) => row.name),
      datasets: [
        { label: "Development WAPE", data: rows.map((row) => n(row.development_WAPE)), backgroundColor: "#64748b" },
        { label: "Recent benchmark WAPE", data: rows.map((row) => n(row.benchmark_WAPE)), backgroundColor: "#f59e0b" },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, ticks: { callback: (value) => `${(value * 100).toFixed(0)}%` } } },
      plugins: { legend: { position: "bottom" } },
    },
  });

  const winner = overnight.recommendation?.winner?.name || "control";
  document.getElementById("overnight-conclusion").innerHTML = `
    <strong>Standalone decision: ${esc(winner)}.</strong>
    <span>The diagnostic autoencoders predicted difficult periods, but their signal did not robustly improve the full NeuralNet as a replacement. The near-winning statistical model improved the recent benchmark and bias, which motivated testing it as a bounded specialist in weekend-v2.</span>
  `;
}

function renderWeekend(data) {
  const weekend = data.weekend_v2?.available ? data.weekend_v2 : data.weekend_v2_smoke;
  const full = data.weekend_v2?.available ? data.weekend_v2 : null;
  const badge = document.getElementById("weekend-status-badge");
  badge.textContent = full ? full.state.replaceAll("_", " ") : "Awaiting full run";
  if (full?.state === "complete") badge.classList.add("winner-badge-good");

  const stages = weekend?.stages || {};
  document.getElementById("weekend-stage-grid").innerHTML = ["screen", "refine", "confirmation"].map((name, index) => {
    const stage = stages[name] || {};
    const expected = stage.expected ?? "?";
    const completed = stage.completed ?? 0;
    const progress = typeof expected === "number" && expected > 0 ? Math.min(completed / expected, 1) : 0;
    return `<article>
      <span class="method-index">0${index + 1}</span>
      <h3>${name[0].toUpperCase()}${name.slice(1)}</h3>
      <strong>${completed} / ${expected}</strong>
      <div class="progress-track"><span style="width:${progress * 100}%"></span></div>
      <small>${stage.failed || 0} failed trials</small>
    </article>`;
  }).join("") + `<article>
    <span class="method-index">04</span><h3>Meta-policy</h3>
    <strong>${weekend?.recommendation ? "Complete" : "Pending"}</strong>
    <div class="progress-track"><span style="width:${weekend?.recommendation ? 100 : 0}%"></span></div>
    <small>Cross-fitted blend and specialist gates</small>
  </article>`;

  const preflight = data.weekend_v2_preflight?.global_convex_crossfit;
  const gate = data.weekend_v2_preflight?.statistical_specialist_gate_max_alpha_0_85;
  document.getElementById("weekend-preflight-note").innerHTML = preflight ? `
    <strong>Why anomaly mode is still present:</strong>
    <span>The prior confirmed experts were complementary. Leave-one-origin-out blending improved development by ${signedPct(preflight.development_relative_improvement)} and benchmark by ${signedPct(preflight.benchmark_relative_improvement)}, with ${ratioPct(preflight.bootstrap_probability_positive, 1)} bootstrap probability of positive gain. A bounded statistical gate improved benchmark by ${signedPct(gate?.benchmark_relative_improvement)}.</span>
  ` : '<strong>Weekend-v2 preflight not found.</strong><span>The full search can still run, but the dashboard cannot show the prior ensemble evidence.</span>';

  const candidateRows = weekend?.confirmation_leaderboard?.length
    ? weekend.confirmation_leaderboard
    : weekend?.refine_leaderboard?.length
      ? weekend.refine_leaderboard
      : weekend?.screen_leaderboard || [];
  document.querySelector("#weekend-candidate-table tbody").innerHTML = candidateRows.map((row) => `
    <tr><td>${esc(row.name)}</td><td>${esc(row.family)}</td><td>${wape(row.development_WAPE)}</td>
      <td>${wape(row.benchmark_WAPE)}</td><td>${signedPct(row.development_relative_improvement)}</td>
      <td>${signedPct(row.benchmark_relative_improvement)}</td></tr>
  `).join("") || '<tr><td colspan="6">The full search has not written a leaderboard yet.</td></tr>';

  const ensembleRows = weekend?.ensemble_leaderboard || [];
  document.querySelector("#weekend-ensemble-table tbody").innerHTML = ensembleRows.map((row) => `
    <tr><td>${esc(row.name)}</td><td>${signedPct(row.development_relative_improvement)}</td>
      <td>${signedPct(row.benchmark_relative_improvement)}</td><td>${ratioPct(row.bootstrap_probability, 1)}</td>
      <td>${row.accepted ? '<span class="tag tag-good">Accepted</span>' : 'No'}</td></tr>
  `).join("") || '<tr><td colspan="5">Meta-policy evaluation runs after confirmation.</td></tr>';

  const command = weekend?.recommendation?.final_submission_command;
  const commandBox = document.getElementById("weekend-final-command");
  if (command) {
    commandBox.hidden = false;
    commandBox.querySelector("code").textContent = command;
  } else {
    commandBox.hidden = true;
  }
}

function renderUnavailable(error) {
  document.getElementById("app").innerHTML = `
    <section class="panel error-panel"><h2>Anomaly dashboard could not load</h2><p>${esc(error.message || error)}</p></section>
  `;
}

async function initAnomalyLab() {
  try {
    wireSharedLinks();
    const [forecast, anomaly] = await Promise.all([loadResults(), loadAnomalyLab()]);
    anomalyPayload = anomaly;
    renderNav(forecast, "anomalies");
    updateStrategyCopy(forecast, canonicalStrategy(forecast));
    renderDecision(anomaly);
    renderKpis(anomaly);
    renderProductOptions(anomaly);
    renderSystemic(anomaly);
    renderAutoencoderSelector(anomaly);
    renderContext(anomaly);
    renderOvernight(anomaly);
    renderWeekend(anomaly);
    const products = anomaly.audit?.products || [];
    if (products.length) await loadProduct(products[0]);
    document.getElementById("anomaly-product-select").addEventListener("change", (event) => loadProduct(event.target.value));
    document.getElementById("anomaly-range-select").addEventListener("change", renderProduct);
    document.getElementById("autoencoder-run-select").addEventListener("change", (event) => renderAutoencoderRun(Number(event.target.value)));

    window.setInterval(async () => {
      try {
        const latest = await loadAnomalyStatus();
        const merged = {
          ...anomalyPayload,
          overnight: latest.overnight,
          weekend_v2: latest.weekend_v2,
          weekend_v2_smoke: latest.weekend_v2_smoke,
        };
        anomalyPayload = merged;
        renderDecision(merged);
        renderWeekend(merged);
      } catch (refreshError) {
        console.warn("Could not refresh search status", refreshError);
      }
    }, 60000);
  } catch (error) {
    console.error(error);
    renderUnavailable(error);
  }
}

document.addEventListener("DOMContentLoaded", initAnomalyLab);
