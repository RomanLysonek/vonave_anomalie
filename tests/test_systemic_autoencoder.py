from __future__ import annotations

import numpy as np
import pandas as pd

from systemic_autoencoder import AutoencoderConfig, fit_score_systemic_autoencoder


def _raw() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    rng = np.random.default_rng(21)
    rows = []
    for product_id, level in enumerate([10.0, 20.0, 30.0, 40.0], start=1):
        quantity = level + 2 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7)
        quantity = np.maximum(0.0, quantity + rng.normal(0, 0.5, len(dates)))
        quantity[-3] *= 5.0
        for date, value in zip(dates, quantity):
            rows.append({
                "ProductId": product_id,
                "DateKey": date,
                "ProductAvailable": True,
                "Quantity": value,
                "QuantityApp": value * 0.2,
                "QuantityWeb": value * 0.8,
            })
    return pd.DataFrame(rows)


def test_autoencoder_scores_windows_and_surfaces_joint_shock():
    cfg = AutoencoderConfig(
        window=14,
        hidden_dim=32,
        latent_dim=6,
        epochs=4,
        batch_size=32,
        train_fraction=0.60,
        calibration_fraction=0.20,
    )
    scores, metadata, model = fit_score_systemic_autoencoder(_raw(), cfg)
    assert len(scores) == 120 - 14 + 1
    assert np.isfinite(scores["systemic_autoencoder_score"]).all()
    assert metadata["n_products"] == 4
    assert metadata["n_windows"] == len(scores)
    shock_date = pd.Timestamp("2025-04-28")
    shock_score = scores.loc[scores["DateKey"] == shock_date, "systemic_autoencoder_score"].iloc[0]
    assert shock_score > scores["systemic_autoencoder_score"].median()
