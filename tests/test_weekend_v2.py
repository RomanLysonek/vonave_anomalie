from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from anomaly_detection import build_demand_anomaly_profile
from anomaly_search_common import (
    apply_candidate_config,
    selected_forecasting_config,
    summarize_oof,
)
from artifact_provenance import (
    artifact_fingerprint,
    config_hash,
    neural_training_identity,
    output_fingerprints,
    result_body_manifest,
)
from framework import Config
from run_weekend_v2_final import (
    _load_resumable_member,
    _member_fingerprint,
    _save_member_cache,
    _safe_relative_path,
    _validate_recommendation,
    active_members,
)
from run_weekend_v2_search import (
    _write_recommendation_with_provenance,
    rank_development_results,
    select_development_winner,
)
from weekend_v2_common import (
    WEEKEND_V2_PROFILES,
    apply_specialist_gate,
    apply_weight_plan,
    build_meta_features,
    crossfit_plan,
    crossfit_specialist_gate,
    generate_weekend_v2_candidates,
    save_pickle,
    search_convex_weights,
    wape,
)


def _authenticated_result(
    root: Path, candidate: dict, *, prediction: float = 9.0
) -> dict:
    trial = root / "confirmation" / candidate["id"]
    trial.mkdir(parents=True)
    origin = pd.Timestamp("2026-01-01")
    oof = pd.DataFrame({
        "ProductId": [1],
        "DateKey": [origin + pd.Timedelta(days=1)],
        "origin": [origin],
        "horizon": [1],
        "actual": [10.0],
        "pred_DynamicRidge": [prediction],
        "ProductAvailable": [True],
    })
    for split in ("development", "benchmark"):
        oof.to_parquet(trial / f"{split}_oof.parquet", index=False)
    fingerprint = artifact_fingerprint(
        schema_version="weekend-source-test-v1",
        semantic={"candidate": candidate},
        source_paths=(),
    )
    payload = {
        "schema_version": "anomaly-forecast-trial-v3",
        "candidate": candidate,
        "model": "DynamicRidge",
        "epochs": 1,
        "seeds": [42],
        "development_origins": ["2026-01-01"],
        "benchmark_origins": ["2026-01-01"],
        "development": summarize_oof(oof, "DynamicRidge"),
        "benchmark": summarize_oof(oof, "DynamicRidge"),
        "status": "complete",
    }
    payload["artifact_manifest"] = {
        "fingerprint": fingerprint,
        "outputs": output_fingerprints(
            trial, ("development_oof.parquet", "benchmark_oof.parquet")
        ),
        "result_body": result_body_manifest(payload),
    }
    result_path = trial / "result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    payload["_result_path"] = str(result_path)
    return payload


def _ensemble_frame() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    start = pd.Timestamp("2024-01-01")
    for origin_index in range(8):
        origin = start + pd.Timedelta(days=28 * origin_index)
        for product in range(1, 6):
            for horizon in range(1, 8):
                actual = 20.0 + product + horizon
                direction = 1.0 if (product + horizon) % 2 == 0 else -1.0
                rows.append(
                    {
                        "ProductId": product,
                        "horizon": horizon,
                        "DateKey": origin + pd.Timedelta(days=horizon),
                        "origin": origin,
                        "origin_type": "development",
                        "validation_stratum": "holiday_event" if origin_index % 4 == 0 else "ordinary",
                        "ProductAvailable": True,
                        "actual": actual,
                        "baseline": actual + 1.0,
                        "pred_SeasonalNaive": actual + 2.0,
                        "pred_MovingAvg28": actual - 2.0,
                        "member__control": actual + 4.0 * direction,
                        "member__specialist": actual - 4.0 * direction,
                        "feature__specialist__anomaly_score_lag1": float(product % 2),
                    }
                )
    return pd.DataFrame(rows), ["member__control", "member__specialist"]


def test_convex_search_finds_complementary_specialists() -> None:
    frame, members = _ensemble_frame()
    fit = search_convex_weights(
        frame,
        members,
        samples=3000,
        seed=7,
        reference_column="member__control",
    )
    weights = fit["weights"]
    assert np.isclose(sum(weights.values()), 1.0)
    prediction = apply_weight_plan(
        frame, {"method": "global_convex", "weights": weights}, members
    )
    assert wape(frame["actual"], prediction) < 1e-8


def test_crossfit_and_product_plan_are_finite() -> None:
    frame, members = _ensemble_frame()
    for method in ("global_convex", "horizon_convex", "product_convex"):
        prediction, plan = crossfit_plan(
            frame,
            members,
            method=method,
            samples=1200,
            seed=17,
            reference_column="member__control",
        )
        assert np.isfinite(prediction).all()
        assert plan["method"] == method
        assert wape(frame["actual"], prediction) < wape(
            frame["actual"], frame["member__control"]
        )


def test_specialist_gate_is_crossfit_safe_and_serializable_shape() -> None:
    frame, _ = _ensemble_frame()
    prediction, bundle = crossfit_specialist_gate(
        frame,
        control_column="member__control",
        specialist_column="member__specialist",
        seed=123,
        max_alpha=0.75,
    )
    assert np.isfinite(prediction).all()
    assert bundle["kind"] == "specialist_gate"
    applied = apply_specialist_gate(frame, bundle)
    assert len(applied) == len(frame)
    assert np.isfinite(applied).all()


def test_meta_features_exclude_target_and_include_prefixed_anomaly_state() -> None:
    frame, members = _ensemble_frame()
    features = build_meta_features(frame, members)
    assert "actual" not in features.columns
    assert "feature__specialist__anomaly_score_lag1" in features.columns
    assert any(column.startswith("product_is_") for column in features.columns)
    assert any(column.startswith("horizon_is_") for column in features.columns)


def test_candidate_generator_is_valid_and_keeps_anomaly_and_non_anomaly_experts() -> None:
    prior = Path("outputs/overnight_anomaly_search")
    candidates = generate_weekend_v2_candidates(
        WEEKEND_V2_PROFILES["smoke"], seed=20260716, prior_root=prior
    )
    families = {item["family"] for item in candidates}
    assert "control" in families
    assert "statistical" in families
    assert "regime" in families
    assert "autoencoder" not in families
    assert "hybrid" not in families
    for item in candidates:
        config = item["config"]
        if "anomaly_rolling_window" in config:
            assert config["anomaly_min_history"] <= config["anomaly_rolling_window"]
    statistical = [item for item in candidates if item["family"] == "statistical"]
    assert statistical
    assert all(
        item["config"]["anomaly_rolling_window"] == 180
        for item in statistical
    )
    assert all(
        item["config"]["anomaly_rolling_window"] != 90
        for item in candidates
        if item["family"] == "statistical"
    )


def test_anomaly_weight_policies_can_downweight_or_emphasize_hard_examples() -> None:
    dates = pd.date_range("2023-01-01", periods=220, freq="D")
    rows = []
    for product in (1, 2, 3):
        for index, date in enumerate(dates):
            quantity = 10.0 + product + (index % 7)
            if index == 180:
                quantity = 180.0 + 10 * product
            if index == 195:
                quantity = 0.0
            rows.append(
                {
                    "ProductId": product,
                    "DateKey": date,
                    "ProductAvailable": True,
                    "Quantity": quantity,
                    "IsSaleOrPromo": False,
                    "CampaignSubTypeWeb": -1,
                    "CampaignSubTypeApp": -1,
                    "DiscountValueWebRelative": 0.0,
                    "DiscountValueAppRelative": 0.0,
                }
            )
    raw = pd.DataFrame(rows)

    down_cfg = Config()
    down_cfg.anomaly_rolling_window = 60
    down_cfg.anomaly_min_history = 21
    down_cfg.anomaly_evt_alpha = 0.10
    down_cfg.anomaly_evt_tail_quantile = 0.80
    down_cfg.anomaly_evt_min_exceedances = 5
    down_cfg.anomaly_weight_strength = 0.5
    down_cfg.anomaly_min_weight = 0.5
    down_cfg.anomaly_weight_policy = "downweight"
    down, _ = build_demand_anomaly_profile(raw, down_cfg)

    hard_cfg = Config(**down_cfg.__dict__)
    hard_cfg.anomaly_weight_policy = "hard_example"
    hard_cfg.anomaly_max_weight = 1.75
    hard, _ = build_demand_anomaly_profile(raw, hard_cfg)

    assert float(down["anomaly_weight"].min()) < 1.0
    assert float(hard["anomaly_weight"].max()) > 1.0
    assert float(hard["anomaly_weight"].max()) <= 1.75 + 1e-12


def test_final_member_resolution_handles_new_plan_types() -> None:
    members = [
        {"column": "member__control", "candidate": {"id": "control"}},
        {"column": "member__stat", "candidate": {"id": "stat"}},
    ]
    specialist = {
        "winner": {
            "plan": {
                "method": "specialist_gate",
                "control_column": "member__control",
                "specialist_column": "member__stat",
            }
        },
        "members": members,
    }
    assert active_members(specialist) == {"member__control", "member__stat"}

    product = {
        "winner": {
            "plan": {
                "method": "product_convex",
                "global_weights": {"member__control": 1.0, "member__stat": 0.0},
                "product_weights": {
                    "1": {"member__control": 0.5, "member__stat": 0.5}
                },
            }
        },
        "members": members,
    }
    assert active_members(product) == {"member__control", "member__stat"}


def test_final_member_resume_requires_matching_manifest_and_csv(tmp_path) -> None:
    train = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Quantity": [2.0, 3.0],
    })
    test = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": pd.to_datetime(["2026-01-03", "2026-01-04"]),
    })
    candidate = {"id": "candidate", "config": {"anomaly_mode": "off"}}
    identity_cfg = apply_candidate_config(selected_forecasting_config(), candidate)
    identity = neural_training_identity(identity_cfg)
    execution = [{
        "seed": 42,
        "device": identity["device"],
        "backend": identity["resolved_backend"],
        "batch_size": identity["batch_size"],
        "reference_batch_size": identity["reference_batch_size"],
    }]
    fingerprint = _member_fingerprint(
        candidate, train, test, epochs=3, seeds=(42,), device="cpu"
    )
    cached = test.copy()
    cached["prediction"] = [4.0, 5.0]
    _save_member_cache(tmp_path, cached, fingerprint, execution)

    loaded = _load_resumable_member(
        tmp_path, fingerprint, test, identity, (42,)
    )
    assert loaded is not None
    assert loaded["prediction"].tolist() == [4.0, 5.0]

    manifest_path = tmp_path / "predictions.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["neural_execution"]["rows"][0]["backend"] = "unrelated"
    manifest["neural_execution"]["sha256"] = config_hash(
        manifest["neural_execution"]["rows"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="execution identity"):
        _load_resumable_member(
            tmp_path, fingerprint, test, identity, (42,)
        )
    _save_member_cache(tmp_path, cached, fingerprint, execution)

    changed_candidate = {"id": "candidate", "config": {"anomaly_mode": "features"}}
    stale = _member_fingerprint(
        changed_candidate, train, test, epochs=3, seeds=(42,), device="cpu"
    )
    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _load_resumable_member(tmp_path, stale, test, identity, (42,))
    assert _load_resumable_member(
        tmp_path,
        stale,
        test,
        identity,
        (42,),
        confirm_recompute_stale=True,
    ) is None

    changed_test = test.assign(Price=[1.0, 2.0])
    changed_fingerprint = _member_fingerprint(
        candidate, train, changed_test, epochs=3, seeds=(42,), device="cpu"
    )
    assert changed_fingerprint != fingerprint
    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _load_resumable_member(
            tmp_path, changed_fingerprint, test, identity, (42,)
        )

    (tmp_path / "predictions.csv").write_text(
        "ProductId,DateKey,prediction\n1,2026-01-03,999\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _load_resumable_member(tmp_path, fingerprint, test, identity, (42,))


def test_weekend_selection_is_invariant_to_frozen_benchmark() -> None:
    def payload(candidate_id: str, family: str, dev: float, bench: float) -> dict:
        summary = {
            "global": {"WAPE": dev},
            "top_actual_decile": {"WAPE": dev},
            "by_origin": {
                "2026-01-01": {"WAPE": dev},
                "2026-02-01": {"WAPE": dev},
            },
        }
        return {
            "candidate": {"id": candidate_id, "name": candidate_id, "family": family},
            "development": summary,
            "benchmark": {"global": {"WAPE": bench}},
        }

    results = [
        payload("control", "control", 1.0, 1.0),
        payload("better", "statistical", 0.8, 1000.0),
        payload("worse", "regime", 0.9, 0.001),
    ]
    before = [row["candidate_id"] for row in rank_development_results(results)]
    results[1]["benchmark"]["global"]["WAPE"] = 0.00001
    results[2]["benchmark"]["global"]["WAPE"] = 99999.0
    after = [row["candidate_id"] for row in rank_development_results(results)]
    assert before == after

    rows = [
        {
            "name": "better",
            "accepted": True,
            "selection_score": 0.2,
            "benchmark": {"relative_improvement": -999.0},
        },
        {
            "name": "worse",
            "accepted": True,
            "selection_score": 0.1,
            "benchmark": {"relative_improvement": 999.0},
        },
    ]
    assert select_development_winner(rows)["name"] == "better"


def test_recommendation_and_required_pickle_tampering_is_rejected(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"artifact_fingerprint": {"input": "bound"}}),
        encoding="utf-8",
    )
    model_path = tmp_path / "ensemble" / "gate.pkl"
    model_path.parent.mkdir()
    member_column = "member__control"
    save_pickle(model_path, {
        "kind": "specialist_gate",
        "members": [member_column, member_column],
    })
    candidate = {"id": "control", "name": "control", "family": "control", "config": {}}
    recommendation = {
        "schema_version": "weekend-v2-search-v4",
        "members": [{"column": member_column, "candidate": candidate}],
        "winner": {
            "name": "gate",
            "plan": {
                "method": "specialist_gate",
                "model_path": "ensemble/gate.pkl",
                "control_column": member_column,
                "specialist_column": member_column,
            },
        },
    }
    results = [_authenticated_result(tmp_path, candidate)]
    _write_recommendation_with_provenance(tmp_path, recommendation, results)
    recommendation_path = tmp_path / "recommendation.json"
    loaded = json.loads(recommendation_path.read_text(encoding="utf-8"))
    _validate_recommendation(recommendation_path, loaded)

    loaded["winner"]["name"] = "tampered"
    recommendation_path.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Recommendation JSON hash"):
        _validate_recommendation(recommendation_path, loaded)

    _write_recommendation_with_provenance(tmp_path, recommendation, results)
    loaded = json.loads(recommendation_path.read_text(encoding="utf-8"))
    model_path.write_bytes(b"tampered-model")
    with pytest.raises(RuntimeError, match="pickle hash mismatch"):
        _validate_recommendation(recommendation_path, loaded)


def test_recommendation_source_missing_tampered_oof_and_traversal_are_rejected(
    tmp_path,
) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"artifact_fingerprint": {"input": "bound"}}),
        encoding="utf-8",
    )
    candidate = {"id": "control", "name": "control", "family": "control", "config": {}}
    result = _authenticated_result(tmp_path, candidate)
    recommendation = {
        "schema_version": "weekend-v2-search-v4",
        "reference_member": "member__control",
        "members": [{"column": "member__control", "candidate": candidate}],
        "winner": {
            "name": "control",
            "plan": {"method": "control", "member": "member__control"},
        },
    }
    _write_recommendation_with_provenance(tmp_path, recommendation, [result])
    recommendation_path = tmp_path / "recommendation.json"
    loaded = json.loads(recommendation_path.read_text(encoding="utf-8"))
    _validate_recommendation(recommendation_path, loaded)

    source_path = Path(result["_result_path"])
    source_bytes = source_path.read_bytes()
    source_path.unlink()
    with pytest.raises(RuntimeError, match="source result is invalid"):
        _validate_recommendation(recommendation_path, loaded)
    source_path.write_bytes(source_bytes)

    oof_path = source_path.parent / "development_oof.parquet"
    oof_bytes = oof_path.read_bytes()
    oof_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="source result is invalid"):
        _validate_recommendation(recommendation_path, loaded)
    oof_path.write_bytes(oof_bytes)

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["development"]["global"]["WAPE"] = 999.0
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source result is invalid"):
        _validate_recommendation(recommendation_path, loaded)

    with pytest.raises(RuntimeError, match="Unsafe provenance path"):
        _safe_relative_path(tmp_path, "../outside/result.json")


def test_legacy_recommendation_requires_future_provenance(tmp_path) -> None:
    path = tmp_path / "recommendation.json"
    legacy = {"winner": {"plan": {"method": "control"}}}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy/unverifiable"):
        _validate_recommendation(path, legacy)
