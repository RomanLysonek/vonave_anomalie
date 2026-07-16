from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly_detection import (
    ANOMALY_ORIGIN_FEATURES,
    apply_anomaly_weights_to_panel,
    attach_anomaly_origin_features,
    build_demand_anomaly_profile,
    calibrate_evt_threshold,
)
from framework import Config


def _history(*, event_spike: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=260, freq="D")
    frames = []
    rng = np.random.default_rng(7)
    for product_id, level in [(1, 20.0), (2, 35.0), (3, 12.0)]:
        weekday = 2.0 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7.0)
        quantity = np.maximum(0.0, level + weekday + rng.normal(0, 1.2, len(dates)))
        if product_id == 1:
            quantity[-1] = 170.0
        frame = pd.DataFrame({
            "ProductId": product_id,
            "DateKey": dates,
            "ProductAvailable": True,
            "Quantity": quantity,
            "QuantityApp": quantity * 0.3,
            "QuantityWeb": quantity * 0.7,
            "IsSaleOrPromo": False,
            "CampaignSubTypeWeb": -1,
            "CampaignSubTypeApp": -1,
            "DiscountValueWebRelative": 0.0,
            "DiscountValueAppRelative": 0.0,
        })
        if product_id == 1 and event_spike:
            frame.loc[frame.index[-1], "IsSaleOrPromo"] = True
            frame.loc[frame.index[-1], "CampaignSubTypeWeb"] = 3
            frame.loc[frame.index[-1], "DiscountValueWebRelative"] = 0.20
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _cfg() -> Config:
    cfg = Config()
    cfg.anomaly_mode = "both"
    cfg.anomaly_min_history = 21
    cfg.anomaly_rolling_window = 120
    cfg.anomaly_evt_min_exceedances = 12
    cfg.anomaly_evt_alpha = 0.02
    cfg.anomaly_min_weight = 0.15
    cfg.anomaly_known_event_min_weight = 0.70
    return cfg


def test_evt_calibration_returns_finite_upper_tail_threshold():
    rng = np.random.default_rng(11)
    scores = np.concatenate([rng.exponential(1.0, 1200), rng.exponential(5.0, 30)])
    result = calibrate_evt_threshold(
        scores,
        alpha=0.01,
        tail_quantile=0.90,
        min_exceedances=20,
    )
    assert np.isfinite(result.threshold)
    assert result.threshold >= np.quantile(scores, 0.80)
    assert result.method in {"evt_pot_gpd", "empirical_quantile_fallback"}


def test_unexplained_spike_is_scored_and_downweighted():
    profile, metadata = build_demand_anomaly_profile(_history(), _cfg())
    spike = profile[(profile["ProductId"] == 1)].iloc[-1]
    assert metadata["n_scored"] > 300
    assert spike["anomaly_score"] >= metadata["local_evt"]["threshold"]
    assert bool(spike["anomaly_flag"])
    assert spike["anomaly_weight"] < 0.70


def test_known_campaign_spike_is_protected_by_weight_floor():
    profile, _ = build_demand_anomaly_profile(_history(event_spike=True), _cfg())
    spike = profile[(profile["ProductId"] == 1)].iloc[-1]
    assert bool(spike["known_event"])
    assert spike["anomaly_weight"] >= 0.70


def test_origin_features_and_panel_weights_are_key_aligned_and_mean_one():
    cfg = _cfg()
    raw = _history()
    profile, _ = build_demand_anomaly_profile(raw, cfg)
    feature_frame = raw[["ProductId", "DateKey"]].copy()
    attached = attach_anomaly_origin_features(feature_frame, profile)
    assert not attached.duplicated(["ProductId", "DateKey"]).any()
    assert set(ANOMALY_ORIGIN_FEATURES).issubset(attached.columns)

    target_dates = raw.groupby("ProductId", sort=False).tail(40)[["ProductId", "DateKey"]]
    panel = target_dates.rename(columns={"DateKey": "TargetDateKey"}).reset_index(drop=True)
    panel["sample_weight"] = np.linspace(0.5, 1.5, len(panel))
    weighted = apply_anomaly_weights_to_panel(panel, profile)
    assert np.isfinite(weighted["sample_weight"]).all()
    assert np.isclose(weighted["sample_weight"].mean(), 1.0)
    assert {"target_anomaly_score", "target_anomaly_flag"}.issubset(weighted.columns)


def test_future_value_does_not_change_earlier_causal_scores():
    cfg = _cfg()
    raw = _history()
    first, _ = build_demand_anomaly_profile(raw, cfg)
    changed = raw.copy()
    mask = (changed["ProductId"] == 1) & (changed["DateKey"] == changed["DateKey"].max())
    changed.loc[mask, "Quantity"] = 10000.0
    second, _ = build_demand_anomaly_profile(changed, cfg)
    cutoff = raw["DateKey"].max() - pd.Timedelta(days=1)
    left = first[first["DateKey"] <= cutoff]["anomaly_score"].to_numpy()
    right = second[second["DateKey"] <= cutoff]["anomaly_score"].to_numpy()
    np.testing.assert_allclose(left, right, equal_nan=True)
