"""Small multivariate demand autoencoder for systemic anomaly diagnostics.

This is the closest analogue to DAVID's sequence autoencoder.  It is kept out
of the default forecasting path because the dataset is small and future sales
are unavailable at prediction time.  Its role is retrospective: identify days
where the joint 30-product demand shape is poorly reconstructed, then compare
those episodes with campaigns, availability changes and forecast failures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anomaly_detection import calibrate_evt_threshold


@dataclass(frozen=True)
class AutoencoderConfig:
    window: int = 28
    hidden_dim: int = 128
    latent_dim: int = 16
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    noise_std: float = 0.02
    train_fraction: float = 0.65
    calibration_fraction: float = 0.20
    evt_alpha: float = 0.02
    seed: int = 42


class DemandWindowAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_daily_matrix(raw_df: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, list[int]]:
    work = raw_df.copy()
    work["DateKey"] = pd.to_datetime(work["DateKey"])
    if "Quantity" not in work:
        work["Quantity"] = (
            pd.to_numeric(work.get("QuantityApp", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(work.get("QuantityWeb", 0.0), errors="coerce").fillna(0.0)
        )
    available = work["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
    work["observed_quantity"] = pd.to_numeric(work["Quantity"], errors="coerce").where(available)
    pivot = work.pivot(index="DateKey", columns="ProductId", values="observed_quantity").sort_index()
    # Missing/unavailable values are imputed with each product's historical
    # median.  Availability anomalies remain visible in the local detector;
    # the autoencoder focuses on the shape of observed demand.
    pivot = pivot.apply(lambda column: column.fillna(column.median()), axis=0).fillna(0.0)
    return pivot.index, np.log1p(np.clip(pivot.to_numpy(dtype=float), 0.0, None)), [
        int(column) for column in pivot.columns
    ]


def _windowize(matrix: np.ndarray, window: int) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if len(matrix) < window:
        raise ValueError("not enough days for requested autoencoder window")
    return np.stack([matrix[end - window:end] for end in range(window, len(matrix) + 1)])


def fit_score_systemic_autoencoder(
    raw_df: pd.DataFrame,
    cfg: AutoencoderConfig = AutoencoderConfig(),
) -> tuple[pd.DataFrame, dict[str, Any], DemandWindowAutoencoder]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dates, matrix, products = build_daily_matrix(raw_df)
    windows_raw = _windowize(matrix, cfg.window)
    n_windows = len(windows_raw)
    train_end = max(1, int(n_windows * cfg.train_fraction))
    calibration_end = max(
        train_end + 1,
        int(n_windows * (cfg.train_fraction + cfg.calibration_fraction)),
    )
    calibration_end = min(calibration_end, n_windows)

    # Fit normalization on autoencoder training windows only.
    train_points = windows_raw[:train_end].reshape(-1, windows_raw.shape[-1])
    mean = np.mean(train_points, axis=0)
    scale = np.std(train_points, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    windows = (windows_raw - mean) / scale
    flat = windows.reshape(n_windows, -1).astype(np.float32)

    device = _device()
    model = DemandWindowAutoencoder(flat.shape[1], cfg.hidden_dim, cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()
    train_tensor = torch.from_numpy(flat[:train_end])
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=min(cfg.batch_size, train_end),
        shuffle=True,
        generator=generator,
    )

    history = []
    model.train()
    for _ in range(cfg.epochs):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            noisy = batch
            if cfg.noise_std > 0.0:
                noisy = batch + cfg.noise_std * torch.randn_like(batch)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(noisy)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))

    model.eval()
    scores = []
    score_loader = DataLoader(TensorDataset(torch.from_numpy(flat)), batch_size=cfg.batch_size)
    with torch.no_grad():
        for (batch,) in score_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            error = torch.mean((reconstructed - batch) ** 2, dim=1)
            scores.append(error.cpu().numpy())
    reconstruction_score = np.concatenate(scores)

    calibration_scores = reconstruction_score[train_end:calibration_end]
    if calibration_scores.size < 20:
        calibration_scores = reconstruction_score[:calibration_end]
    evt = calibrate_evt_threshold(
        calibration_scores,
        alpha=cfg.evt_alpha,
        tail_quantile=0.85,
        min_exceedances=max(8, min(20, calibration_scores.size // 4)),
    )
    window_end_dates = dates[cfg.window - 1:]
    result = pd.DataFrame({
        "DateKey": window_end_dates,
        "systemic_autoencoder_score": reconstruction_score,
        "systemic_autoencoder_flag": reconstruction_score >= evt.threshold,
        "autoencoder_split": np.where(
            np.arange(n_windows) < train_end,
            "train",
            np.where(np.arange(n_windows) < calibration_end, "calibration", "holdout"),
        ),
    })
    metadata = {
        "config": asdict(cfg),
        "device": str(device),
        "n_products": len(products),
        "products": products,
        "n_windows": n_windows,
        "train_windows": train_end,
        "calibration_windows": calibration_end - train_end,
        "holdout_windows": n_windows - calibration_end,
        "final_train_loss": history[-1],
        "evt": evt.to_dict(),
        "n_flagged": int(result["systemic_autoencoder_flag"].sum()),
    }
    return result, metadata, model
