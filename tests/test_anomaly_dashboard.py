from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

from anomaly_dashboard import build_anomaly_dashboard, build_product_payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_anomaly_dashboard_aggregates_audit_and_search_status(tmp_path: Path) -> None:
    audit = tmp_path / "outputs" / "anomaly_audit_real"
    audit.mkdir(parents=True)
    _write_json(audit / "anomaly_metadata.json", {
        "n_local_anomalies": 1,
        "n_systemic_days": 1,
        "n_scored": 2,
        "local_evt": {"threshold": 3.0},
        "test_context": {"n_rows": 1, "n_shift_flags": 0},
    })
    pd.DataFrame([
        {
            "ProductId": 1, "DateKey": "2026-01-01", "Quantity": 10,
            "expected_quantity": 9, "anomaly_signed_residual": 0.1,
            "anomaly_score": 1.0, "anomaly_flag": False,
            "anomaly_rate_28": 0.0, "days_since_anomaly": None,
            "known_event": False, "systemic_anomaly_score": 1.0,
            "systemic_anomaly_flag": False, "systemic_anomaly_rate_28": 0.0,
            "anomaly_weight": 1.0,
        },
        {
            "ProductId": 1, "DateKey": "2026-01-02", "Quantity": 40,
            "expected_quantity": 10, "anomaly_signed_residual": 1.3,
            "anomaly_score": 5.0, "anomaly_flag": True,
            "anomaly_rate_28": 0.5, "days_since_anomaly": 0,
            "known_event": True, "systemic_anomaly_score": 4.0,
            "systemic_anomaly_flag": True, "systemic_anomaly_rate_28": 0.5,
            "anomaly_weight": 0.9,
        },
    ]).to_csv(audit / "demand_anomaly_profile.csv", index=False)
    pd.DataFrame([{
        "ProductId": 1, "DateKey": "2026-01-12", "context_risk_raw": 0.2,
        "context_risk_percentile": 0.8, "context_shift_flag": False,
    }]).to_csv(audit / "test_context_risk.csv", index=False)
    pd.DataFrame([{
        "DateKey": "2026-01-12", "mean_context_risk": 0.5,
        "max_context_risk": 0.8, "shifted_products": 0,
    }]).to_csv(audit / "test_context_risk_daily.csv", index=False)

    weekend = tmp_path / "outputs" / "weekend_v2_search"
    _write_json(weekend / "manifest.json", {"candidate_count": 4})
    _write_json(weekend / "refine_candidates.json", [{}, {}])
    _write_json(weekend / "screen" / "one" / "result.json", {"status": "complete"})

    payload = build_anomaly_dashboard(tmp_path)

    assert payload["audit"]["available"] is True
    assert payload["audit"]["event_protected_anomalies"] == 1
    assert payload["audit"]["product_summary"][0]["local_anomalies"] == 1
    assert payload["audit"]["daily"][1]["systemic_flag"] is True
    assert payload["weekend_v2"]["state"] == "running"
    assert payload["weekend_v2"]["stages"]["screen"] == {
        "completed": 1, "failed": 0, "expected": 4,
    }


def test_product_payload_returns_timeline_and_context(tmp_path: Path) -> None:
    audit = tmp_path / "outputs" / "anomaly_audit_real"
    audit.mkdir(parents=True)
    rows = [{
        "ProductId": 7, "DateKey": "2026-01-01", "Quantity": 20,
        "expected_quantity": 10, "anomaly_signed_residual": 0.65,
        "anomaly_score": 4.5, "anomaly_flag": True,
        "anomaly_rate_28": 0.1, "days_since_anomaly": 0,
        "known_event": False, "systemic_anomaly_score": 2.0,
        "systemic_anomaly_flag": False, "systemic_anomaly_rate_28": 0.0,
        "anomaly_weight": 0.8,
    }]
    pd.DataFrame(rows).to_csv(audit / "demand_anomaly_profile.csv", index=False)
    pd.DataFrame([{
        "ProductId": 7, "DateKey": "2026-01-12", "context_risk_raw": 0.3,
        "context_risk_percentile": 0.9, "context_shift_flag": False,
    }]).to_csv(audit / "test_context_risk.csv", index=False)

    payload = build_product_payload(tmp_path, 7)

    assert payload["available"] is True
    assert payload["summary"]["local_anomalies"] == 1
    assert payload["timeline"][0]["DateKey"] == "2026-01-01"
    assert payload["future_context"][0]["context_risk_percentile"] == 0.9


def test_product_payload_reports_missing_product(tmp_path: Path) -> None:
    audit = tmp_path / "outputs" / "anomaly_audit_real"
    audit.mkdir(parents=True)
    pd.DataFrame([{
        "ProductId": 1, "DateKey": "2026-01-01", "Quantity": 1,
        "expected_quantity": 1, "anomaly_signed_residual": 0,
        "anomaly_score": 0, "anomaly_flag": False,
        "anomaly_rate_28": 0, "days_since_anomaly": None,
        "known_event": False, "systemic_anomaly_score": 0,
        "systemic_anomaly_flag": False, "systemic_anomaly_rate_28": 0,
        "anomaly_weight": 1,
    }]).to_csv(audit / "demand_anomaly_profile.csv", index=False)

    payload = build_product_payload(tmp_path, 99)

    assert payload["available"] is False
    assert "does not exist" in payload["message"]


def test_anomaly_lab_has_research_description_strip() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "webapp" / "static"
    html = (static_dir / "anomalies.html").read_text(encoding="utf-8")

    assert 'class="model-hero page-description-strip" style="--mc:#f59e0b"' in html
    assert "Baseline models remain in the Overview comparison" in html
    assert "Control NeuralNet" in (static_dir / "common.js").read_text(encoding="utf-8")



def test_retained_pages_share_identical_utility_and_site_header_markup() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "webapp" / "static"
    pages = ["index.html", "anomalies.html", "dataset.html", "evaluation.html", "model.html"]

    def fragment(html: str, start: str, end: str) -> str:
        return html[html.index(start):html.index(end, html.index(start)) + len(end)]

    utility_fragments = []
    header_fragments = []
    for page in pages:
        html = (static_dir / page).read_text(encoding="utf-8")
        utility_fragments.append(fragment(html, '<div class="promo-bar">', "</div>"))
        header_fragments.append(fragment(html, '<header class="hero">', "</header>"))
        assert "styles.css?v=22" in html

    assert len(set(utility_fragments)) == 1
    assert len(set(header_fragments)) == 1


def test_anomaly_description_strip_uses_shared_model_hero_geometry() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "webapp" / "static"
    html = (static_dir / "anomalies.html").read_text(encoding="utf-8")
    styles = (static_dir / "styles.css").read_text(encoding="utf-8")
    script = (static_dir / "anomalies.js").read_text(encoding="utf-8")

    assert '<header class="model-hero page-description-strip" style="--mc:#f59e0b">' in html
    assert "anomaly-hero-strip" not in html
    assert ".anomaly-hero-strip" not in styles
    assert ".anomaly-site-hero" not in styles
    assert "updateStrategyCopy(forecast, canonicalStrategy(forecast));" in script
    assert "promo-anomaly-status" not in html
    assert "promo-anomaly-status" not in script


def test_retained_detail_pages_share_explicit_strip_width_contract() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "webapp" / "static"
    pages = ["anomalies.html", "dataset.html", "evaluation.html", "model.html"]
    for page in pages:
        html = (static_dir / page).read_text(encoding="utf-8")
        assert "model-hero page-description-strip" in html

    styles = (static_dir / "styles.css").read_text(encoding="utf-8")
    assert "--dashboard-shell-width: 1280px;" in styles
    assert ".page-description-strip {" in styles
    assert "max-width: var(--dashboard-shell-width);" in styles
    assert "scrollbar-gutter: stable;" in styles
