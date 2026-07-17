"use strict";

let anomalyPayload = null;
let productPayload = null;
let productChart = null;
let systemicChart = null;
let contextChart = null;
let productRequestToken = 0;

async function loadPublishedJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status})`);
  return response.json();
}

function anomalyDataPath() {
  return window.STATIC_DASHBOARD
    ? "./data/anomaly-dashboard-v2.json"
    : "/data/anomaly-dashboard-v2.json";
}

function productDataPath(productId) {
  const relative = `data/anomaly-products-v2/product-${encodeURIComponent(productId)}.json`;
  return window.STATIC_DASHBOARD ? `./${relative}` : `/${relative}`;
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function number(value, digits = 1) {
  if (value === null || value === undefined || value === "") return "—";
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  return parsed.toLocaleString("en-GB", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function percent(value, digits = 1, signed = false) {
  if (value === null || value === undefined || value === "") return "—";
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  const formatted = (parsed * 100).toLocaleString("en-GB", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
  return `${signed && parsed >= 0 ? "+" : ""}${formatted}%`;
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isSha256(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function isNullableFinite(value) {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

function validateAggregatePayload(data) {
  if (!isObject(data) || data.schema_version !== "anomaly-dashboard-v2") {
    throw new Error("Published anomaly aggregate has an unsupported schema.");
  }
  if (!isSha256(data.source_manifest_hash)) {
    throw new Error("Published anomaly aggregate has no valid source manifest hash.");
  }
  const audit = data.audit;
  if (
    !isObject(audit)
    || audit.available !== true
    || !Array.isArray(audit.products)
    || audit.products.length === 0
    || !Array.isArray(audit.product_summary)
    || !Array.isArray(audit.daily)
    || !Array.isArray(audit.top_exceedances)
    || !Array.isArray(audit.context_daily)
  ) {
    throw new Error("Published anomaly aggregate violates the audit schema.");
  }
  const ids = audit.products;
  if (ids.some((id) => !Number.isInteger(id) || id < 1) || new Set(ids).size !== ids.length) {
    throw new Error("Published anomaly aggregate contains invalid product IDs.");
  }
  const summaryIds = audit.product_summary.map((row) => row?.ProductId);
  if (
    audit.product_summary.length !== ids.length
    || summaryIds.some((id) => !Number.isInteger(id))
    || !ids.every((id) => summaryIds.includes(id))
    || audit.product_summary.some((row) =>
      !isObject(row)
      || !Number.isInteger(row.observed_days)
      || !Number.isInteger(row.threshold_exceedances)
      || !isNullableFinite(row.max_anomaly_score)
      || !isNullableFinite(row.mean_training_weight))
    || audit.daily.some((row) =>
      !isObject(row)
      || typeof row.DateKey !== "string"
      || !isNullableFinite(row.total_quantity)
      || !isNullableFinite(row.threshold_exceedances))
    || audit.top_exceedances.some((row) => !isObject(row))
    || audit.context_daily.some((row) => !isObject(row))
  ) {
    throw new Error("Published anomaly aggregate contains invalid audit rows.");
  }
  return data;
}

function validateProductPayload(data, selectedProductId) {
  const selected = Number(selectedProductId);
  if (!Number.isInteger(selected) || !anomalyPayload?.audit?.products?.map(Number).includes(selected)) {
    throw new Error(`Selected product ID ${selectedProductId} is not in the canonical aggregate.`);
  }
  if (
    !isObject(data)
    || data.schema_version !== "anomaly-product-v2"
    || data.available !== true
    || data.product_id !== selected
    || data.source_manifest_hash !== anomalyPayload.source_manifest_hash
    || !isObject(data.summary)
    || !Array.isArray(data.timeline)
    || !Array.isArray(data.top_exceedances)
    || !Array.isArray(data.future_context)
  ) {
    throw new Error(`Published anomaly artifact for Product ${selected} failed schema or provenance validation.`);
  }
  if (
    !Number.isInteger(data.summary.observed_days)
    || !Number.isInteger(data.summary.threshold_exceedances)
    || !Number.isInteger(data.summary.known_event_exceedances)
    || !isNullableFinite(data.summary.max_score)
    || !isNullableFinite(data.summary.mean_training_weight)
    || data.timeline.some((row) =>
      !isObject(row)
      || typeof row.DateKey !== "string"
      || !isNullableFinite(row.Quantity)
      || !isNullableFinite(row.expected_quantity)
      || !isNullableFinite(row.anomaly_score)
      || typeof row.anomaly_flag !== "boolean")
    || data.top_exceedances.some((row) => !isObject(row))
    || data.future_context.some((row) => !isObject(row))
  ) {
    throw new Error(`Published anomaly artifact for Product ${selected} contains invalid rows.`);
  }
  return data;
}

function renderBanner(data) {
  const banner = data.snapshot_banner || {};
  document.getElementById("snapshot-banner").innerHTML = `
    <strong>${esc(banner.version || data.schema_version)}</strong>
    <span>Snapshot through ${esc(banner.snapshot_as_of || data.snapshot_as_of)} · ${esc(banner.provenance || "provenance unavailable")}</span>
    <code title="Source manifest hash">${esc(String(data.source_manifest_hash || "").slice(0, 16))}</code>
  `;
}

function renderDecision(data) {
  const recommendation = data.recommendation || {};
  const policy = recommendation.policy || "unavailable";
  const mode = recommendation.anomaly_mode || "unavailable";
  document.getElementById("anomaly-decision-title").textContent =
    `${policy === "control" ? "Control retained" : policy} · anomaly_mode=${mode}`;
  document.getElementById("anomaly-decision-copy").textContent =
    recommendation.reason || "Standalone anomaly policies did not beat the control.";
  document.querySelector("#anomaly-decision-card .status-dot").className =
    "status-dot status-neutral";
}

function renderKpis(data) {
  const audit = data.audit || {};
  const metadata = audit.metadata || {};
  const local = metadata.local_evt || {};
  const systemic = metadata.systemic_evt || {};
  const context = metadata.test_context || {};
  const cards = [
    ["Product-day exceedances", number(metadata.n_local_anomalies, 0), `${percent(local.validation_exceedance_rate, 2)} validation exceedance rate`],
    ["Systemic review days", number(metadata.n_systemic_days, 0), `${percent(systemic.validation_exceedance_rate, 2)} validation exceedance rate`],
    ["Known-event overlaps", number(audit.known_event_exceedances, 0), "Explanatory proxies, not labels"],
    ["Context shift flags", number(context.n_shift_flags, 0), `${number(context.n_rows, 0)} future-known rows reviewed`],
    ["V2 autoencoder", data.autoencoder_v2?.available ? "Verified" : "Unavailable", "Leakage-safe V2 is canonical"],
    ["Weekend-v2", "Not run", "No tuning on frozen periods"],
  ];
  document.getElementById("anomaly-kpi-grid").innerHTML = cards.map(([label, value, note]) => `
    <div class="kpi-card anomaly-kpi-card">
      <p class="kpi-label">${esc(label)}</p><p class="kpi-value">${esc(value)}</p><p class="kpi-sub">${esc(note)}</p>
    </div>
  `).join("");
}

function renderProductOptions(data) {
  const select = document.getElementById("anomaly-product-select");
  const summaries = data.audit?.product_summary || [];
  select.innerHTML = (data.audit?.products || []).map((productId) => {
    const summary = summaries.find((row) => Number(row.ProductId) === Number(productId));
    return `<option value="${productId}">Product ${productId} · ${number(summary?.threshold_exceedances, 0)} exceedances</option>`;
  }).join("");
}

function productWindowRows(rows) {
  const range = document.getElementById("anomaly-range-select").value;
  return range === "all" ? rows : rows.slice(-Number(range));
}

function chartOptions(leftTitle, rightTitle) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { ticks: { maxTicksLimit: 12 }, grid: { color: "#eeeeee" } },
      y: { position: "left", beginAtZero: true, title: { display: true, text: leftTitle } },
      y1: { position: "right", beginAtZero: true, title: { display: true, text: rightTitle }, grid: { drawOnChartArea: false } },
    },
    plugins: { legend: { position: "bottom" } },
  };
}

function renderProduct() {
  if (!productPayload?.available) return;
  const summary = productPayload.summary || {};
  document.getElementById("anomaly-product-summary").innerHTML = [
    ["Observed days", number(summary.observed_days, 0)],
    ["Threshold exceedances", number(summary.threshold_exceedances, 0)],
    ["Known-event overlaps", number(summary.known_event_exceedances, 0)],
    ["Maximum score", number(summary.max_score, 2)],
    ["Mean training weight", number(summary.mean_training_weight, 3)],
  ].map(([label, value]) => `<span><small>${esc(label)}</small><strong>${esc(value)}</strong></span>`).join("");

  const rows = productWindowRows(productPayload.timeline || []);
  if (productChart) productChart.destroy();
  productChart = new Chart(document.getElementById("chart-product-anomalies"), {
    type: "line",
    data: {
      labels: rows.map((row) => row.DateKey),
      datasets: [
        { label: "Observed", data: rows.map((row) => row.Quantity), borderColor: "#0a0a0a", pointRadius: 0, borderWidth: 1.8, yAxisID: "y" },
        { label: "Causal expectation", data: rows.map((row) => row.expected_quantity), borderColor: "#64748b", pointRadius: 0, borderDash: [5, 4], yAxisID: "y" },
        { label: "Score", data: rows.map((row) => row.anomaly_score), borderColor: "#b76600", pointRadius: 0, yAxisID: "y1" },
        { label: "Exceedance", data: rows.map((row) => row.anomaly_flag ? row.Quantity : null), borderColor: "#b76600", backgroundColor: "#b76600", showLine: false, pointRadius: 4, yAxisID: "y" },
      ],
    },
    options: chartOptions("Quantity", "Score"),
  });

  document.querySelector("#product-anomaly-table tbody").innerHTML =
    (productPayload.top_exceedances || []).map((row) => `
      <tr><td>${esc(row.DateKey)}</td><td>${number(row.Quantity)}</td><td>${number(row.expected_quantity)}</td>
      <td><strong>${number(row.anomaly_score, 2)}</strong></td>
      <td>${row.known_event ? "Known event proxy" : "No known event proxy"}</td><td>${number(row.anomaly_weight, 3)}</td></tr>
    `).join("") || '<tr><td colspan="6">No threshold exceedances for this product.</td></tr>';
}

function clearProduct(error, selectedProductId) {
  productPayload = null;
  if (productChart) {
    productChart.destroy();
    productChart = null;
  }
  const label = `Product ${esc(selectedProductId)}`;
  document.getElementById("anomaly-product-summary").innerHTML =
    `<span class="error-panel"><strong>${label} unavailable</strong><small>${esc(error.message || error)}</small></span>`;
  document.querySelector("#product-anomaly-table tbody").innerHTML =
    `<tr><td colspan="6">${label} could not be loaded or validated.</td></tr>`;
}

async function loadProduct(productId) {
  const requestToken = ++productRequestToken;
  clearProduct(new Error("Loading selected product…"), productId);
  try {
    const payload = await loadPublishedJson(productDataPath(productId));
    if (requestToken !== productRequestToken) return false;
    productPayload = validateProductPayload(payload, productId);
    renderProduct();
    return true;
  } catch (error) {
    if (requestToken !== productRequestToken) return false;
    console.error(error);
    clearProduct(error, productId);
    return false;
  }
}

function renderSystemic(data) {
  const rows = data.audit?.daily || [];
  if (systemicChart) systemicChart.destroy();
  systemicChart = new Chart(document.getElementById("chart-systemic-timeline"), {
    type: "line",
    data: {
      labels: rows.map((row) => row.DateKey),
      datasets: [
        { label: "Total quantity", data: rows.map((row) => row.total_quantity), borderColor: "#0a0a0a", pointRadius: 0, yAxisID: "y" },
        { label: "Threshold exceedances", data: rows.map((row) => row.threshold_exceedances), borderColor: "#b76600", pointRadius: 0, yAxisID: "y1" },
      ],
    },
    options: chartOptions("Daily quantity", "Review signals"),
  });
  document.querySelector("#top-local-table tbody").innerHTML =
    (data.audit?.top_exceedances || []).map((row) => `
      <tr><td>${esc(row.ProductId)}</td><td>${esc(row.DateKey)}</td><td>${number(row.Quantity)}</td>
      <td>${number(row.expected_quantity)}</td><td><strong>${number(row.anomaly_score, 2)}</strong></td>
      <td>${row.known_event ? "Known event proxy" : row.systemic_anomaly_flag ? "Systemic overlap" : "Product only"}</td></tr>
    `).join("");
}

function renderContext(data) {
  const rows = data.audit?.context_daily || [];
  if (contextChart) contextChart.destroy();
  contextChart = new Chart(document.getElementById("chart-context-risk"), {
    type: "line",
    data: {
      labels: rows.map((row) => row.DateKey),
      datasets: [
        { label: "Mean novelty percentile", data: rows.map((row) => row.mean_context_risk), borderColor: "#0a0a0a" },
        { label: "Maximum novelty percentile", data: rows.map((row) => row.max_context_risk), borderColor: "#b76600" },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 0, max: 1 } } },
  });
  const context = data.audit?.metadata?.test_context || {};
  document.getElementById("context-risk-note").innerHTML =
    `<strong>${number(context.n_shift_flags, 0)} of ${number(context.n_rows, 0)} rows crossed the context-shift threshold.</strong>
    <span>This diagnostic uses future-known commercial context and no future demand.</span>`;
}

function renderEvidence(data) {
  const v2 = data.autoencoder_v2 || {};
  const overnight = data.overnight || {};
  const weekend = data.weekend_v2 || {};
  document.getElementById("evidence-list").innerHTML = [
    ["V2 autoencoder", v2.state, v2.message],
    ["Overnight search", overnight.state, overnight.message],
    ["Weekend-v2", weekend.state, weekend.message],
    ["Compact autoencoder", "excluded", "Legacy compact evidence is excluded because preprocessing used future-derived medians."],
  ].map(([name, state, message]) => `
    <article><div><strong>${esc(name)}</strong><span>${esc(String(state || "unknown").replaceAll("_", " "))}</span></div><p>${esc(message)}</p></article>
  `).join("");
}

function renderPreflight(data) {
  const preflight = data.weekend_v2_preflight || {};
  const interval = preflight.bootstrap_ci_95 || [];
  document.getElementById("weekend-preflight-note").innerHTML = `
    <strong>${percent(preflight.development_relative_improvement, 2, true)} development change · non-nested preflight</strong>
    <span>${esc(preflight.message)} CI [${percent(interval[0], 2, true)}, ${percent(interval[1], 2, true)}]. Weekend-v2 was not run.</span>
  `;
}

function renderUnavailable(error) {
  document.getElementById("app").innerHTML =
    `<section class="panel error-panel"><h2>Published anomaly snapshot unavailable</h2><p>${esc(error.message || error)}</p></section>`;
}

async function initAnomalyLab() {
  try {
    const [forecast, anomaly] = await Promise.all([loadResults(), loadPublishedJson(anomalyDataPath())]);
    anomalyPayload = validateAggregatePayload(anomaly);
    renderNav(forecast, "anomalies");
    updateStrategyCopy(forecast, canonicalStrategy(forecast));
    renderBanner(anomaly);
    renderDecision(anomaly);
    renderKpis(anomaly);
    renderProductOptions(anomaly);
    renderSystemic(anomaly);
    renderContext(anomaly);
    renderEvidence(anomaly);
    renderPreflight(anomaly);
    const firstProduct = anomaly.audit?.products?.[0];
    if (firstProduct) await loadProduct(firstProduct);
    document.getElementById("anomaly-product-select").addEventListener("change", (event) => loadProduct(event.target.value));
    document.getElementById("anomaly-range-select").addEventListener("change", renderProduct);
  } catch (error) {
    console.error(error);
    renderUnavailable(error);
  }
}

document.addEventListener("DOMContentLoaded", initAnomalyLab);
