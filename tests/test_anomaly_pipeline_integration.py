from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly_detection import (
    ANOMALY_ORIGIN_FEATURES,
    apply_anomaly_weights_to_panel,
    attach_anomaly_origin_features,
    build_demand_anomaly_profile,
)
from framework import (
    Config,
    add_train_lags,
    build_direct_panel,
    direct_panel_feature_names,
    prepare_features,
    product_reference_dates,
    select_trainable_panel_rows,
)


def _raw() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=420, freq="D")
    rng = np.random.default_rng(9)
    frames = []
    for product_id, level, price in [(1, 15.0, 120.0), (2, 30.0, 240.0)]:
        quantity = np.maximum(
            0.0,
            level + 3 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7)
            + rng.normal(0, 1.0, len(dates)),
        )
        frame = pd.DataFrame({
            "ProductId": product_id,
            "DateKey": dates,
            "ProductAvailable": True,
            "Quantity": quantity,
            "QuantityApp": quantity * 0.25,
            "QuantityWeb": quantity * 0.75,
            "IsSaleOrPromo": False,
            "CampaignSubTypeWeb": -1,
            "CampaignSubTypeApp": -1,
            "DiscountValueWebRelative": 0.0,
            "DiscountValueAppRelative": 0.0,
            "PriceLocalVat": price,
            "is_gap_filled": False,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_anomaly_layer_flows_into_direct_panel_without_target_feature_leakage():
    raw = _raw()
    cfg = Config()
    cfg.anomaly_mode = "both"
    cfg.anomaly_min_history = 21
    cfg.anomaly_evt_min_exceedances = 12

    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(raw)
    features = prepare_features(raw, price_ref, first_seen, first_available, cfg)
    features = add_train_lags(features, cfg.lag_windows)
    profile, _ = build_demand_anomaly_profile(raw, cfg)
    features = attach_anomaly_origin_features(features, profile)

    panel = build_direct_panel(features, range(1, 8), cfg=cfg)
    selected = select_trainable_panel_rows(
        panel,
        cutoff=raw["DateKey"].max(),
        available_only=True,
        cfg=cfg,
    )
    weighted = apply_anomaly_weights_to_panel(selected, profile)

    feature_names = direct_panel_feature_names(cfg)
    assert set(ANOMALY_ORIGIN_FEATURES).issubset(feature_names)
    assert set(ANOMALY_ORIGIN_FEATURES).issubset(weighted.columns)
    assert "target_anomaly_score" not in feature_names
    assert "target_anomaly_flag" not in feature_names
    assert np.isfinite(weighted["sample_weight"]).all()
    assert np.isclose(weighted["sample_weight"].mean(), 1.0)
