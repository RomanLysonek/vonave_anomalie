from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "webapp" / "static" / "anomalies.js"


def test_anomaly_client_enforces_aggregate_and_product_provenance_contracts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'data.schema_version !== "anomaly-dashboard-v2"' in source
    assert 'data.schema_version !== "anomaly-product-v2"' in source
    assert "isSha256(data.source_manifest_hash)" in source
    assert "data.product_id !== selected" in source
    assert "data.source_manifest_hash !== anomalyPayload.source_manifest_hash" in source
    assert "Selected product ID" in source


def test_anomaly_client_keeps_null_unavailable_and_clears_failed_selection() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'value === null || value === undefined || value === ""' in source
    assert "productPayload = null;" in source
    assert "productChart.destroy();" in source
    assert "could not be loaded or validated" in source
    assert "clearProduct(error, productId)" in source


def test_anomaly_client_ignores_out_of_order_product_responses() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "let productRequestToken = 0;" in source
    assert "const requestToken = ++productRequestToken;" in source
    assert source.count("requestToken !== productRequestToken") == 2
