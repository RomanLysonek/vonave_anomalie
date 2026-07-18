"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const staticDir = path.join(root, "webapp", "static");

function source(name) {
  return fs.readFileSync(path.join(staticDir, name), "utf8");
}

function checkSyntax() {
  for (const name of ["common.js", "app.js", "model.js", "evaluation.js", "dataset.js", "anomalies.js"]) {
    const result = spawnSync(process.execPath, ["--check", path.join(staticDir, name)], {
      encoding: "utf8",
    });
    assert.strictEqual(result.status, 0, `${name} syntax check failed:\n${result.stderr}`);
  }
}

function checkIndependentNavigation() {
  const nav = { innerHTML: "" };
  const context = {
    window: { STATIC_DASHBOARD: true },
    console,
    document: {
      getElementById: (id) => id === "site-nav" ? nav : null,
      querySelectorAll: () => [],
    },
  };
  vm.createContext(context);
  vm.runInContext(source("common.js"), context);
  context.renderNav({}, "dataset");

  for (const label of ["Anomaly overview", "Data & transfer", "Evaluation", "Control forecast"]) {
    assert.ok(nav.innerHTML.includes(label), label);
  }
  assert.ok(nav.innerHTML.includes('href="index.html"'));
  assert.ok(nav.innerHTML.includes('href="dataset.html"'));
  assert.ok(nav.innerHTML.includes('href="evaluation.html"'));
  assert.ok(nav.innerHTML.includes('href="model.html"'));
  assert.ok(!nav.innerHTML.includes("Classical" + " Forecasting"));
  assert.ok(!nav.innerHTML.includes("Chronos-2" + " Challenger"));
  assert.ok(nav.innerHTML.includes("nav-pill active"));
}

function checkStaticAnomalyContract() {
  const anomalySource = source("anomalies.js");
  assert.ok(anomalySource.includes("./data/anomaly-dashboard-v2.json"));
  assert.ok(anomalySource.includes("data/anomaly-products-v2/product-"));
  assert.ok(anomalySource.includes('toLocaleString("en-GB"'));
  assert.ok(!anomalySource.includes("/api/anomaly-lab"));
  assert.ok(!anomalySource.includes("setInterval"));

  const html = source("index.html");
  assert.ok(html.includes("DAVID / DBAAS knowledge transfer"));
  assert.ok(html.includes("No adjudicated anomaly labels exist"));
  assert.ok(html.includes("anomaly_mode=off"));
  assert.ok(!html.includes('id="strategy-select"'));
}

checkSyntax();
checkIndependentNavigation();
checkStaticAnomalyContract();
console.log("3 JavaScript smoke checks passed");
