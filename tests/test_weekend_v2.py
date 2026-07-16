from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_detection import build_demand_anomaly_profile
from framework import Config
from run_weekend_v2_final import active_members
from weekend_v2_common import (
    WEEKEND_V2_PROFILES,
    apply_specialist_gate,
    apply_weight_plan,
    build_meta_features,
    crossfit_plan,
    crossfit_specialist_gate,
    generate_weekend_v2_candidates,
    search_convex_weights,
    wape,
)


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
    for item in candidates:
        config = item["config"]
        if "anomaly_rolling_window" in config:
            assert config["anomaly_min_history"] <= config["anomaly_rolling_window"]


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
