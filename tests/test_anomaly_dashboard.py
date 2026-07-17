from __future__ import annotations

import json
from pathlib import Path

import pytest

from webapp.anomaly_dashboard import (
    SCHEMA_VERSION,
    _v2_evidence,
    _validate_preflight,
    _validate_weight_ablation,
    build_anomaly_dashboard,
    build_product_payload,
    publish_anomaly_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_dashboard_is_deterministic_and_honest(tmp_path: Path) -> None:
    first = build_anomaly_dashboard(ROOT)
    second = build_anomaly_dashboard(ROOT)

    assert first == second
    assert first["schema_version"] == SCHEMA_VERSION
    assert first["recommendation"] == {
        "policy": "control",
        "anomaly_mode": "off",
        "status": "current",
        "verified_evidence": {
            "artifact": "outputs/real_data_test_summary.json",
            "weight_ablation_winner": "control",
        },
        "reason": (
            "The verified weighting ablation selected control, and no "
            "provenance-complete checked-in evidence justifies anomaly promotion."
        ),
    }
    assert first["generated_at"] is None
    assert first["snapshot_as_of"] == "2026-01-18"
    assert first["research_status"]["truth_labels_available"] is False
    assert first["autoencoder_v2"]["available"] is False
    assert first["weekend_v2"]["state"] == "not_run"
    assert first["weekend_v2_preflight"]["state"] == "contaminated"
    assert first["weekend_v2_preflight"]["provenance"] == "unverified"
    assert first["weekend_v2_preflight"]["selection_use"] == "excluded"
    assert first["weekend_v2_preflight"]["development_relative_improvement"] is None
    assert first["overnight"]["scientific_status"] == "contaminated"
    assert len(first["excluded_evidence"]) == 2
    assert all(item["sha256"] for item in first["excluded_evidence"])
    assert first["audit"]["products"] == list(range(1, 31))

    publish_anomaly_artifacts(ROOT, tmp_path)
    aggregate = json.loads((tmp_path / "anomaly-dashboard-v2.json").read_text())
    assert aggregate == first
    checked_in = json.loads(
        (ROOT / "docs" / "data" / "anomaly-dashboard-v2.json").read_text()
    )
    assert checked_in == first
    products = sorted((tmp_path / "anomaly-products-v2").glob("product-*.json"))
    assert len(products) == 30


def test_publish_removes_stale_product_artifacts(tmp_path: Path) -> None:
    stale = tmp_path / "anomaly-products-v2"
    stale.mkdir(parents=True)
    (stale / "product-31.json").write_text("{}")

    publish_anomaly_artifacts(ROOT, tmp_path)

    product_ids = sorted(
        int(path.stem.removeprefix("product-"))
        for path in (tmp_path / "anomaly-products-v2").glob("product-*.json")
    )
    assert product_ids == list(range(1, 31))


def test_reproducible_build_time_does_not_change_snapshot_date(monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    payload = build_anomaly_dashboard(ROOT)

    assert payload["generated_at"] == "1970-01-01T00:00:00Z"
    assert payload["snapshot_as_of"] == "2026-01-18"


def test_product_payload_is_versioned_and_has_full_timeline() -> None:
    payload = build_product_payload(ROOT, 1)
    assert payload["schema_version"] == "anomaly-product-v2"
    assert payload["available"] is True
    assert payload["product_id"] == 1
    assert payload["timeline"]
    assert payload["timeline"][0]["DateKey"] == "2021-01-01"
    assert payload["future_context"]
    assert "threshold_exceedances" in payload["summary"]


def test_product_payload_reports_missing_product() -> None:
    payload = build_product_payload(ROOT, 99)
    assert payload["available"] is False
    assert "does not exist" in payload["message"]


def test_static_anomaly_client_uses_generated_relative_data_without_polling() -> None:
    script = (ROOT / "webapp" / "static" / "anomalies.js").read_text()
    assert "data/anomaly-dashboard-v2.json" in script
    assert "data/anomaly-products-v2/product-" in script
    assert "/api/anomaly-lab" not in script
    assert "setInterval" not in script
    assert 'toLocaleString("en-GB"' in script


def test_optional_api_returns_exact_generated_files(tmp_path: Path, monkeypatch) -> None:
    from webapp import server

    aggregate = {"schema_version": SCHEMA_VERSION, "exact": [1, 2, 3]}
    aggregate_path = tmp_path / "anomaly-dashboard-v2.json"
    aggregate_path.write_text(json.dumps(aggregate))
    products = tmp_path / "products"
    products.mkdir()
    product = {"schema_version": "anomaly-product-v2", "product_id": 7}
    (products / "product-7.json").write_text(json.dumps(product))
    monkeypatch.setattr(server, "ANOMALY_DASHBOARD_PATH", aggregate_path)
    monkeypatch.setattr(server, "ANOMALY_PRODUCTS_DIR", products)

    assert json.loads(server.get_anomaly_lab().body) == aggregate
    assert json.loads(server.get_anomaly_product(7).body) == product


def test_weight_ablation_rejects_empty_and_contradictory_evidence() -> None:
    assert _validate_weight_ablation({"forecast_weight_ablation": {"summary": []}})[0] is None
    payload = json.loads((ROOT / "outputs" / "real_data_test_summary.json").read_text())
    benchmark_rows = [
        row for row in payload["forecast_weight_ablation"]["summary"]
        if row["split"] == "benchmark"
    ]
    benchmark_rows[0]["WAPE"] = 999.0
    benchmark_rows[1]["WAPE"] = 0.0
    assert _validate_weight_ablation(payload)[0]["winner"] == "control"
    payload["forecast_weight_ablation"]["winner"] = "weight_soft"
    evidence, reason = _validate_weight_ablation(payload)
    assert evidence is None
    assert "contradicts" in reason


def test_preflight_derives_improvement_and_crosses_zero_from_metrics() -> None:
    payload = json.loads((ROOT / "reports" / "weekend_v2_preflight.json").read_text())
    evidence, reason = _validate_preflight(payload)
    assert reason == "valid"
    assert evidence is not None
    assert evidence["development_relative_improvement"] == pytest.approx(
        (payload["control"]["development_WAPE"]
         - payload["global_convex_crossfit"]["development_WAPE"])
        / payload["control"]["development_WAPE"]
    )
    assert evidence["confidence_interval_crosses_zero"] is True

    payload["global_convex_crossfit"]["development_relative_improvement"] = 0.99
    assert _validate_preflight(payload)[0] is None


def test_v2_fake_hashes_and_empty_outputs_are_invalid(tmp_path: Path) -> None:
    artifact = tmp_path / "outputs" / "anomaly_autoencoder_v2"
    artifact.mkdir(parents=True)
    (artifact / "artifact_manifest.json").write_text(json.dumps({
        "schema_version": "systemic-autoencoder-v2",
        "canonical_inputs": {"input.csv": {"sha256": "truthy", "size": 1}},
        "configuration": {"window": 28},
        "canonical_outputs": [],
        "outputs": {},
        "fingerprint": {
            "provenance_schema_version": "artifact-provenance-v3",
            "source_hash": "truthy",
            "dependency_manifest_hash": "truthy",
        },
        "body": {"available": True},
    }))

    result = _v2_evidence(tmp_path)
    assert result["available"] is False
    assert result["state"] == "invalid"
