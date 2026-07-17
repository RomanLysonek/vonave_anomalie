import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard_artifacts import (
    collect_ablation_showcase,
    publish_static_dashboard,
    summarize_per_product_oof,
    summarize_top_deciles,
)
from ml.publish_site import _assert_replace_safe


PREDICTIONS = {"NeuralNet": "pred_NeuralNet", "XGBoost": "pred_XGBoost"}


def _oof():
    actual = np.array([10, 20, 80, 100, 15, 25], dtype=float)
    return pd.DataFrame({
        "origin_type": ["development"] * 3 + ["recent_benchmark"] * 3,
        "strategy": "direct",
        "origin": pd.to_datetime(["2024-01-01"] * 3 + ["2025-12-01"] * 3),
        "validation_stratum": ["winter_test_like"] * 3 + ["holiday_event"] * 3,
        "ProductId": [1, 1, 2, 1, 2, 2],
        "DateKey": pd.date_range("2024-01-02", periods=6),
        "horizon": [1, 2, 3, 1, 2, 3],
        "ProductAvailable": True,
        "actual": actual,
        "pred_NeuralNet": actual + np.array([1, -2, 10, -10, 1, 2]),
        "pred_XGBoost": actual + np.array([2, -1, 5, -5, -1, 1]),
    })


def test_product_and_top_decile_diagnostics_are_common_population():
    oof = _oof()
    product = summarize_per_product_oof(oof, PREDICTIONS)
    assert set(product["model"]) == {"NeuralNet", "XGBoost"}
    assert set(product["origin_type"]) == {"development", "recent_benchmark"}
    assert product["WAPE"].notna().all()

    decile, errors = summarize_top_deciles(oof, PREDICTIONS, quantile=0.8, max_error_rows=2)
    assert set(decile["model"]) == {"NeuralNet", "XGBoost"}
    assert (decile["n"] >= 1).all()
    assert len(errors) <= 8
    assert errors["absolute_error"].notna().all()


def test_ablation_showcase_marks_recommendations(tmp_path):
    c1 = tmp_path / "c1_screening"
    c1.mkdir()
    pd.DataFrame([
        {"candidate": "control", "stage": "recency", "model": "NeuralNet", "WAPE": 0.3, "test_aligned_WAPE": 0.31, "BiasRatio": 0.1, "Coverage": 1.0},
        {"candidate": "winner", "stage": "recency", "model": "NeuralNet", "WAPE": 0.28, "test_aligned_WAPE": 0.29, "BiasRatio": 0.05, "Coverage": 1.0},
    ]).to_csv(c1 / "c1_screening_results.csv", index=False)
    (c1 / "recommendation.json").write_text(json.dumps({"recommendation": {"candidate": "winner"}}))
    result = collect_ablation_showcase(tmp_path)
    assert result.loc[result["candidate"].eq("winner"), "selected"].all()


def test_static_dashboard_refuses_unowned_docs_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    generated = docs / "index.html"
    generated.write_text("generated")
    (docs / "site-manifest.json").write_text(json.dumps({
        "files": [{"path": "index.html", "sha256": "not-used-for-ownership"}],
    }))
    (docs / "authored-notes.md").write_text("must survive")

    with pytest.raises(RuntimeError, match="unowned files"):
        _assert_replace_safe(docs)


def test_checked_in_docs_are_generated_and_manifest_owned():
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    manifest = json.loads((docs / "site-manifest.json").read_text())
    owned = {row["path"] for row in manifest["files"]} | {"site-manifest.json"}
    actual = {
        path.relative_to(docs).as_posix()
        for path in docs.rglob("*")
        if path.is_file()
    }
    assert actual == owned
    assert (docs / "data" / "results.json").is_file()
    assert (docs / "data" / "anomaly-dashboard-v2.json").is_file()
    assert len(list((docs / "data" / "anomaly-products-v2").glob("product-*.json"))) == 30


def test_pages_checkout_does_not_persist_credentials():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pages.yml"
    ).read_text()
    assert "persist-credentials: false" in workflow
