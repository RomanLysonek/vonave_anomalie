from __future__ import annotations

import numpy as np
import pandas as pd

from framework import Config, direct_panel_feature_names
from systemic_autoencoder_v2 import (
    AUTOENCODER_ORIGIN_FEATURES,
    AutoencoderV2Config,
    apply_autoencoder_weights_to_panel,
    attach_autoencoder_origin_features,
    fit_score_systemic_autoencoder_v2,
)


def _synthetic_raw(days: int = 140, products: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    for product in range(1, products + 1):
        level = 8.0 + product * 2.0
        for index, date in enumerate(dates):
            weekly = 1.0 + 0.25 * np.sin(2 * np.pi * date.dayofweek / 7)
            quantity = max(0.0, level * weekly + rng.normal(0.0, 1.0))
            event = index == days - 20
            if event:
                quantity *= 5.0
            rows.append(
                {
                    "ProductId": product,
                    "DateKey": date,
                    "ProductAvailable": True,
                    "Quantity": quantity,
                    "QuantityApp": quantity * 0.35,
                    "QuantityWeb": quantity * 0.65,
                    "IsSaleOrPromo": event,
                    "CampaignSubTypeWeb": 3 if event else -1,
                    "CampaignSubTypeApp": -1,
                    "DiscountValueWebRelative": 0.2 if event else 0.0,
                    "DiscountValueAppRelative": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_v2_autoencoder_temporal_profile_and_weights_are_well_formed() -> None:
    raw = _synthetic_raw()
    cfg = AutoencoderV2Config(
        window=7,
        representation="weekday_residual",
        architecture="mlp",
        hidden_dim=16,
        latent_dim=4,
        max_epochs=2,
        patience=1,
        batch_size=32,
        training_window_days=60,
        calibration_days=20,
        holdout_days=20,
        min_train_windows=20,
        threshold_method="empirical",
        score_aggregation="mean",
        seed=7,
        device="cpu",
    )
    profile, metadata, _, preprocessor = fit_score_systemic_autoencoder_v2(raw, cfg)

    assert not profile.empty
    assert set(profile["autoencoder_split"]) >= {
        "train",
        "validation",
        "calibration",
        "holdout",
    }
    assert profile["DateKey"].is_monotonic_increasing
    assert profile["autoencoder_score"].notna().all()
    assert profile["autoencoder_percentile"].between(0.0, 1.0).all()
    assert profile["autoencoder_weight"].between(cfg.min_weight, 1.0).all()
    assert metadata["best_epoch"] >= 1
    assert metadata["epochs_ran"] <= cfg.max_epochs
    assert metadata["split_counts"]["holdout"] >= 20
    assert preprocessor["mean"].shape == preprocessor["scale"].shape


def test_autoencoder_features_and_target_weights_align_by_date() -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    feature_df = pd.DataFrame(
        {
            "ProductId": [1, 2, 1, 2, 1, 2],
            "DateKey": np.repeat(dates, 2),
        }
    )
    daily = pd.DataFrame(
        {
            "DateKey": dates,
            "autoencoder_score": [0.1, 1.0, 2.0],
            "autoencoder_percentile": [0.2, 0.9, 1.0],
            "autoencoder_flag": [False, False, True],
            "autoencoder_weight": [1.0, 0.8, 0.4],
            "autoencoder_score_mean_7": [0.1, 0.55, 1.03],
            "autoencoder_score_mean_28": [0.1, 0.55, 1.03],
            "autoencoder_flag_rate_28": [0.0, 0.0, 1 / 3],
            "days_since_autoencoder_anomaly": [np.nan, np.nan, 0.0],
        }
    )
    attached = attach_autoencoder_origin_features(feature_df, daily)
    assert attached.loc[attached["DateKey"] == dates[-1], "autoencoder_score_lag0"].eq(2.0).all()

    panel = pd.DataFrame(
        {
            "ProductId": [1, 2, 1, 2],
            "TargetDateKey": [dates[1], dates[1], dates[2], dates[2]],
            "sample_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    weighted = apply_autoencoder_weights_to_panel(panel, daily, normalize=False)
    assert weighted["sample_weight"].tolist() == [0.8, 0.8, 0.4, 0.4]


def test_autoencoder_feature_schema_is_source_aware() -> None:
    cfg = Config()
    cfg.anomaly_mode = "features"
    cfg.anomaly_source = "autoencoder"
    schema = direct_panel_feature_names(cfg)
    assert all(column in schema for column in AUTOENCODER_ORIGIN_FEATURES)
    assert "anomaly_score_lag0" not in schema

    cfg.anomaly_source = "hybrid"
    hybrid = direct_panel_feature_names(cfg)
    assert all(column in hybrid for column in AUTOENCODER_ORIGIN_FEATURES)
    assert "anomaly_score_lag0" in hybrid
