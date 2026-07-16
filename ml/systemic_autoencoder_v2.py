"""GPU-capable, temporally calibrated systemic demand autoencoders.

The original ``systemic_autoencoder.py`` is intentionally small and useful for
one diagnostic run.  This module is the experiment-grade implementation used
by the overnight search:

* train/validation/calibration/holdout are chronological and disjoint;
* all imputers and scalers are fitted on the training period only;
* multiple demand representations and architectures are supported;
* early stopping restores the best temporal-validation checkpoint;
* calibration produces continuous percentiles, flags and bounded weights;
* profiles are cacheable per fold/configuration so expensive candidates can be
  reused by several downstream forecasting actions.

The autoencoder remains systemic: one score describes the joint state of all
products for a window ending on a given date.  The downstream forecasting code
therefore attaches the same origin-state score to every product on that date.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import gc
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anomaly_detection import calibrate_evt_threshold


AUTOENCODER_ORIGIN_FEATURES = [
    "autoencoder_score_lag0",
    "autoencoder_percentile_lag0",
    "autoencoder_flag_lag0",
    "autoencoder_score_mean_7",
    "autoencoder_score_mean_28",
    "autoencoder_flag_rate_28",
    "days_since_autoencoder_anomaly",
]


@dataclass(frozen=True)
class AutoencoderV2Config:
    window: int = 28
    representation: Literal[
        "log_level",
        "weekday_residual",
        "level_residual",
        "residual_availability",
        "level_residual_availability",
    ] = "weekday_residual"
    architecture: Literal["mlp", "conv", "gru"] = "conv"
    hidden_dim: int = 128
    latent_dim: int = 16
    dropout: float = 0.10
    max_epochs: int = 160
    patience: int = 24
    min_delta: float = 1e-4
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    noise_std: float = 0.03
    loss: Literal["mse", "huber"] = "huber"
    huber_delta: float = 1.0
    grad_clip: float = 5.0
    training_window_days: int | None = 1095
    calibration_days: int = 180
    holdout_days: int = 180
    validation_fraction: float = 0.15
    min_train_windows: int = 120
    evt_alpha: float = 0.02
    evt_tail_quantile: float = 0.85
    threshold_method: Literal["evt", "empirical"] = "evt"
    score_aggregation: Literal["mean", "p95", "hybrid"] = "hybrid"
    input_clip: float = 8.0
    weight_strength: float = 0.75
    min_weight: float = 0.50
    known_event_min_weight: float = 0.85
    seed: int = 42
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    num_workers: int = 0


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    holdout: np.ndarray
    ignored: np.ndarray


class MLPAutoencoder(nn.Module):
    def __init__(
        self,
        window: int,
        n_features: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = window * n_features
        self.window = window
        self.n_features = n_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.flatten(start_dim=1)
        reconstructed = self.decoder(self.encoder(flat))
        return reconstructed.reshape(-1, self.window, self.n_features)


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_channels = max(16, min(hidden_dim, 256))
        latent_channels = max(4, min(latent_dim, hidden_channels))
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, n_features, kernel_size=5, padding=2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        channels_first = values.transpose(1, 2)
        return self.decoder(self.encoder(channels_first)).transpose(1, 2)


class GRUAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(values)
        latent = self.to_latent(hidden[-1])
        repeated = latent.unsqueeze(1).expand(-1, values.shape[1], -1)
        initial = torch.tanh(self.latent_to_hidden(latent)).unsqueeze(0)
        decoded, _ = self.decoder(repeated, initial)
        return self.output(decoded)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps":
        available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not available:
            raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _known_event_by_day(raw_df: pd.DataFrame) -> pd.Series:
    index = raw_df.index
    sale = raw_df.get("IsSaleOrPromo", pd.Series(False, index=index))
    sale = sale.astype("boolean").fillna(False).astype(bool)
    cw = pd.to_numeric(
        raw_df.get("CampaignSubTypeWeb", pd.Series(-1, index=index)), errors="coerce"
    ).fillna(-1)
    ca = pd.to_numeric(
        raw_df.get("CampaignSubTypeApp", pd.Series(-1, index=index)), errors="coerce"
    ).fillna(-1)
    dw = pd.to_numeric(
        raw_df.get("DiscountValueWebRelative", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    da = pd.to_numeric(
        raw_df.get("DiscountValueAppRelative", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    event = sale | cw.ne(-1) | ca.ne(-1) | dw.gt(0.0) | da.gt(0.0)
    dates = pd.to_datetime(raw_df["DateKey"])
    return event.groupby(dates).any().sort_index()


def _quantity(raw_df: pd.DataFrame) -> pd.Series:
    if "Quantity" in raw_df:
        return pd.to_numeric(raw_df["Quantity"], errors="coerce")
    app = pd.to_numeric(raw_df.get("QuantityApp", 0.0), errors="coerce").fillna(0.0)
    web = pd.to_numeric(raw_df.get("QuantityWeb", 0.0), errors="coerce").fillna(0.0)
    return app + web


def build_daily_representation(
    raw_df: pd.DataFrame,
    representation: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, list[str], pd.Series]:
    """Create a daily multivariate representation without future imputation.

    Missing quantities remain NaN here.  Imputation and scaling are fitted later
    from the autoencoder training interval only.
    """
    work = raw_df.copy()
    work["DateKey"] = pd.to_datetime(work["DateKey"])
    work = work.sort_values(["ProductId", "DateKey"]).reset_index(drop=True)
    if work.duplicated(["ProductId", "DateKey"]).any():
        raise ValueError("Autoencoder input must be unique by ProductId and DateKey")
    work["Quantity"] = _quantity(work)
    availability = (
        work.get("ProductAvailable", pd.Series(False, index=work.index))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    work["observed_quantity"] = work["Quantity"].where(availability)

    products = sorted(int(value) for value in work["ProductId"].dropna().unique())
    dates = pd.date_range(work["DateKey"].min(), work["DateKey"].max(), freq="D")
    quantity = (
        work.pivot(index="DateKey", columns="ProductId", values="observed_quantity")
        .reindex(index=dates, columns=products)
    )
    available = (
        work.assign(_available=availability.astype(float))
        .pivot(index="DateKey", columns="ProductId", values="_available")
        .reindex(index=dates, columns=products)
        .fillna(0.0)
    )

    level = np.log1p(np.clip(quantity.to_numpy(dtype=float), 0.0, None))
    baseline_parts = []
    baseline_weights = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=float)
    for lag in (7, 14, 21, 28):
        baseline_parts.append(quantity.shift(lag).to_numpy(dtype=float))
    lag_matrix = np.stack(baseline_parts, axis=2)
    observed = np.isfinite(lag_matrix)
    numerator = np.nansum(lag_matrix * baseline_weights[None, None, :], axis=2)
    denominator = np.sum(observed * baseline_weights[None, None, :], axis=2)
    baseline = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    residual = level - np.log1p(np.clip(baseline, 0.0, None))
    availability_matrix = available.to_numpy(dtype=float)

    level_names = [f"level_p{product}" for product in products]
    residual_names = [f"weekday_residual_p{product}" for product in products]
    availability_names = [f"available_p{product}" for product in products]

    if representation == "log_level":
        matrix, names = level, level_names
    elif representation == "weekday_residual":
        matrix, names = residual, residual_names
    elif representation == "level_residual":
        matrix = np.concatenate([level, residual], axis=1)
        names = level_names + residual_names
    elif representation == "residual_availability":
        matrix = np.concatenate([residual, availability_matrix], axis=1)
        names = residual_names + availability_names
    elif representation == "level_residual_availability":
        matrix = np.concatenate([level, residual, availability_matrix], axis=1)
        names = level_names + residual_names + availability_names
    else:
        raise ValueError(f"Unknown autoencoder representation: {representation}")

    event_by_day = _known_event_by_day(work).reindex(dates, fill_value=False)
    return pd.DatetimeIndex(dates), matrix.astype(float), names, event_by_day


def _windowize(matrix: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if window < 2 or len(matrix) < window:
        raise ValueError("Not enough days for requested autoencoder window")
    windows = np.stack([matrix[end - window : end] for end in range(window, len(matrix) + 1)])
    end_positions = np.arange(window - 1, len(matrix), dtype=int)
    return windows, end_positions


def make_temporal_split(
    window_end_dates: pd.DatetimeIndex,
    cfg: AutoencoderV2Config,
) -> TemporalSplit:
    if not 0.0 < cfg.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    if cfg.calibration_days < 20 or cfg.holdout_days < 20:
        raise ValueError("calibration_days and holdout_days must each be at least 20")
    last_date = pd.Timestamp(window_end_dates.max())
    holdout_start = last_date - pd.Timedelta(days=cfg.holdout_days - 1)
    calibration_start = holdout_start - pd.Timedelta(days=cfg.calibration_days)

    holdout = np.flatnonzero(window_end_dates >= holdout_start)
    calibration = np.flatnonzero(
        (window_end_dates >= calibration_start) & (window_end_dates < holdout_start)
    )
    train_pool = np.flatnonzero(window_end_dates < calibration_start)
    if cfg.training_window_days is not None and train_pool.size:
        train_start_date = calibration_start - pd.Timedelta(days=cfg.training_window_days)
        train_pool = train_pool[window_end_dates[train_pool] >= train_start_date]

    min_required = max(20, cfg.min_train_windows)
    if train_pool.size < min_required or calibration.size < 20 or holdout.size < 20:
        # Deterministic fraction fallback for short synthetic tests and early folds.
        n = len(window_end_dates)
        if n < 60:
            raise ValueError(
                f"Only {n} windows; at least 60 are required for temporal splitting"
            )
        train_pool_end = min(
            max(min_required, int(n * 0.65)),
            n - 40,
        )
        calibration_end = min(
            max(train_pool_end + 20, int(n * 0.85)),
            n - 20,
        )
        train_pool = np.arange(0, train_pool_end, dtype=int)
        calibration = np.arange(train_pool_end, calibration_end, dtype=int)
        holdout = np.arange(calibration_end, n, dtype=int)

    n_validation = max(12, int(round(train_pool.size * cfg.validation_fraction)))
    n_validation = min(n_validation, max(1, train_pool.size // 3))
    train = train_pool[:-n_validation]
    validation = train_pool[-n_validation:]
    if train.size < max(10, cfg.min_train_windows // 2):
        raise ValueError(
            f"Only {train.size} autoencoder training windows after temporal splitting"
        )
    used = np.concatenate([train, validation, calibration, holdout])
    ignored = np.setdiff1d(np.arange(len(window_end_dates), dtype=int), used)
    return TemporalSplit(
        train=train,
        validation=validation,
        calibration=calibration,
        holdout=holdout,
        ignored=ignored,
    )


def _fit_preprocessor(
    windows_raw: np.ndarray,
    train_indices: np.ndarray,
    input_clip: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_points = windows_raw[train_indices].reshape(-1, windows_raw.shape[-1])
    median = np.nanmedian(train_points, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(windows_raw), windows_raw, median[None, None, :])
    train_filled = filled[train_indices].reshape(-1, filled.shape[-1])
    mean = np.mean(train_filled, axis=0)
    scale = np.std(train_filled, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    transformed = (filled - mean[None, None, :]) / scale[None, None, :]
    transformed = np.clip(transformed, -float(input_clip), float(input_clip))
    return transformed.astype(np.float32), {
        "median": median.astype(np.float32),
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
    }


def build_model(
    cfg: AutoencoderV2Config,
    n_features: int,
) -> nn.Module:
    if cfg.architecture == "mlp":
        return MLPAutoencoder(
            cfg.window, n_features, cfg.hidden_dim, cfg.latent_dim, cfg.dropout
        )
    if cfg.architecture == "conv":
        return ConvAutoencoder(n_features, cfg.hidden_dim, cfg.latent_dim, cfg.dropout)
    if cfg.architecture == "gru":
        return GRUAutoencoder(n_features, cfg.hidden_dim, cfg.latent_dim, cfg.dropout)
    raise ValueError(f"Unknown autoencoder architecture: {cfg.architecture}")


def _loss_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: AutoencoderV2Config,
) -> torch.Tensor:
    if cfg.loss == "mse":
        return (prediction - target).square()
    if cfg.loss == "huber":
        return nn.functional.huber_loss(
            prediction, target, delta=float(cfg.huber_delta), reduction="none"
        )
    raise ValueError(f"Unknown autoencoder loss: {cfg.loss}")


def _mean_loader_loss(
    model: nn.Module,
    values: torch.Tensor,
    batch_size: int,
    device: torch.device,
    cfg: AutoencoderV2Config,
) -> float:
    loader = DataLoader(
        TensorDataset(values),
        batch_size=min(batch_size, len(values)),
        shuffle=False,
        num_workers=0,
    )
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            loss = _loss_values(model(batch), batch, cfg).mean()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _score_model(
    model: nn.Module,
    values: torch.Tensor,
    cfg: AutoencoderV2Config,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(values),
        batch_size=min(cfg.batch_size, len(values)),
        shuffle=False,
        num_workers=0,
    )
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            element = (reconstructed - batch).square().flatten(start_dim=1)
            mean = element.mean(dim=1)
            if cfg.score_aggregation == "mean":
                score = mean
            else:
                p95 = torch.quantile(element, 0.95, dim=1)
                score = p95 if cfg.score_aggregation == "p95" else 0.5 * mean + 0.5 * p95
            scores.append(score.detach().cpu().numpy())
    return np.concatenate(scores).astype(float)


def _empirical_percentile(scores: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    reference = np.sort(calibration_scores[np.isfinite(calibration_scores)])
    if reference.size == 0:
        return np.full(len(scores), np.nan, dtype=float)
    return np.searchsorted(reference, scores, side="right") / reference.size


def _days_since_flag(flag: pd.Series) -> pd.Series:
    result = np.full(len(flag), np.nan, dtype=float)
    last: int | None = None
    for idx, value in enumerate(flag.fillna(False).astype(bool).to_numpy()):
        if value:
            last = idx
            result[idx] = 0.0
        elif last is not None:
            result[idx] = float(idx - last)
    return pd.Series(result, index=flag.index)


def fit_score_systemic_autoencoder_v2(
    raw_df: pd.DataFrame,
    cfg: AutoencoderV2Config = AutoencoderV2Config(),
) -> tuple[pd.DataFrame, dict[str, Any], nn.Module, dict[str, np.ndarray]]:
    """Fit one temporal autoencoder and return a daily causal-origin profile."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dates, matrix, feature_names, event_by_day = build_daily_representation(
        raw_df, cfg.representation
    )
    windows_raw, end_positions = _windowize(matrix, cfg.window)
    end_dates = pd.DatetimeIndex(dates[end_positions])
    split = make_temporal_split(end_dates, cfg)
    windows, preprocessor = _fit_preprocessor(windows_raw, split.train, cfg.input_clip)

    device = _resolve_device(cfg.device)
    model = build_model(cfg, windows.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.max_epochs), eta_min=cfg.learning_rate * 0.02
    )
    train_tensor = torch.from_numpy(windows[split.train])
    validation_tensor = torch.from_numpy(windows[split.validation])
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=min(cfg.batch_size, len(train_tensor)),
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    train_history: list[float] = []
    validation_history: list[float] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for (batch,) in loader:
            batch = batch.to(device)
            noisy = batch
            if cfg.noise_std > 0.0:
                noisy = batch + cfg.noise_std * torch.randn_like(batch)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(noisy)
            loss = _loss_values(reconstructed, batch, cfg).mean()
            loss.backward()
            if cfg.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_loss = float(np.mean(batch_losses))
        validation_loss = _mean_loader_loss(
            model, validation_tensor, cfg.batch_size, device, cfg
        )
        train_history.append(train_loss)
        validation_history.append(validation_loss)

        if validation_loss < best_loss - cfg.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("Autoencoder training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    all_tensor = torch.from_numpy(windows)
    scores = _score_model(model, all_tensor, cfg, device)
    calibration_scores = scores[split.calibration]

    if cfg.threshold_method == "evt":
        evt = calibrate_evt_threshold(
            calibration_scores,
            alpha=cfg.evt_alpha,
            tail_quantile=cfg.evt_tail_quantile,
            min_exceedances=max(8, min(30, len(calibration_scores) // 5)),
        )
        threshold = float(evt.threshold)
        threshold_metadata = evt.to_dict()
    elif cfg.threshold_method == "empirical":
        threshold = float(np.quantile(calibration_scores, 1.0 - cfg.evt_alpha))
        threshold_metadata = {
            "threshold": threshold,
            "method": "empirical_quantile",
            "alpha": cfg.evt_alpha,
            "tail_quantile": 1.0 - cfg.evt_alpha,
            "n_total": int(len(calibration_scores)),
        }
    else:
        raise ValueError(f"Unknown threshold_method: {cfg.threshold_method}")

    percentile = _empirical_percentile(scores, calibration_scores)
    flag = np.isfinite(scores) & (scores >= threshold)
    split_labels = np.full(len(scores), "ignored", dtype=object)
    split_labels[split.train] = "train"
    split_labels[split.validation] = "validation"
    split_labels[split.calibration] = "calibration"
    split_labels[split.holdout] = "holdout"

    calibration_median = float(np.median(calibration_scores))
    calibration_iqr = float(
        np.quantile(calibration_scores, 0.75) - np.quantile(calibration_scores, 0.25)
    )
    score_scale = max(calibration_iqr / 1.349, 1e-6)
    excess = np.maximum((scores - threshold) / score_scale, 0.0)
    weight = np.exp(-cfg.weight_strength * excess)
    weight = np.clip(weight, cfg.min_weight, 1.0)
    # In-sample reconstruction errors are not used to attenuate supervised rows.
    # Only the post-calibration holdout is strictly out-of-sample for both the
    # autoencoder and threshold. Calibration rows define the threshold and are
    # therefore not attenuated.
    out_of_sample = split_labels == "holdout"
    weight[~out_of_sample] = 1.0

    events = event_by_day.reindex(end_dates, fill_value=False).to_numpy(dtype=bool)
    positive_event_floor = max(cfg.min_weight, cfg.known_event_min_weight)
    weight[events & flag] = np.maximum(weight[events & flag], positive_event_floor)

    result = pd.DataFrame(
        {
            "DateKey": end_dates,
            "autoencoder_score": scores,
            "autoencoder_percentile": percentile,
            "autoencoder_flag": flag,
            "autoencoder_weight": weight,
            "autoencoder_split": split_labels,
            "autoencoder_known_event": events,
        }
    )
    result["autoencoder_score_mean_7"] = result["autoencoder_score"].rolling(
        7, min_periods=1
    ).mean()
    result["autoencoder_score_mean_28"] = result["autoencoder_score"].rolling(
        28, min_periods=1
    ).mean()
    result["autoencoder_flag_rate_28"] = result["autoencoder_flag"].astype(float).rolling(
        28, min_periods=1
    ).mean()
    result["days_since_autoencoder_anomaly"] = _days_since_flag(
        result["autoencoder_flag"]
    )

    holdout_scores = scores[split.holdout]
    holdout_flags = flag[split.holdout]
    calibration_flags = flag[split.calibration]
    metadata = {
        "schema_version": "systemic-autoencoder-v2",
        "config": asdict(cfg),
        "device": str(device),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "n_windows": int(len(scores)),
        "split_counts": {
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "calibration": int(len(split.calibration)),
            "holdout": int(len(split.holdout)),
            "ignored": int(len(split.ignored)),
        },
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(train_history)),
        "best_validation_loss": float(best_loss),
        "final_train_loss": float(train_history[-1]),
        "final_validation_loss": float(validation_history[-1]),
        "train_history": train_history,
        "validation_history": validation_history,
        "threshold": threshold_metadata,
        "calibration_flag_rate": float(np.mean(calibration_flags)),
        "holdout_flag_rate": float(np.mean(holdout_flags)),
        "holdout_score_mean": float(np.mean(holdout_scores)),
        "holdout_score_std": float(np.std(holdout_scores)),
        "calibration_score_median": calibration_median,
        "calibration_score_scale": score_scale,
        "n_flagged": int(np.sum(flag)),
        "n_holdout_flagged": int(np.sum(holdout_flags)),
    }
    return result, metadata, model, preprocessor


def config_from_framework(cfg: Any) -> AutoencoderV2Config:
    """Translate forecasting ``Config`` fields into the V2 dataclass."""
    return AutoencoderV2Config(
        window=int(getattr(cfg, "autoencoder_window", 28)),
        representation=str(getattr(cfg, "autoencoder_representation", "weekday_residual")),
        architecture=str(getattr(cfg, "autoencoder_architecture", "conv")),
        hidden_dim=int(getattr(cfg, "autoencoder_hidden_dim", 128)),
        latent_dim=int(getattr(cfg, "autoencoder_latent_dim", 16)),
        dropout=float(getattr(cfg, "autoencoder_dropout", 0.10)),
        max_epochs=int(getattr(cfg, "autoencoder_max_epochs", 160)),
        patience=int(getattr(cfg, "autoencoder_patience", 24)),
        min_delta=float(getattr(cfg, "autoencoder_min_delta", 1e-4)),
        batch_size=int(getattr(cfg, "autoencoder_batch_size", 64)),
        learning_rate=float(getattr(cfg, "autoencoder_learning_rate", 1e-3)),
        weight_decay=float(getattr(cfg, "autoencoder_weight_decay", 1e-5)),
        noise_std=float(getattr(cfg, "autoencoder_noise_std", 0.03)),
        loss=str(getattr(cfg, "autoencoder_loss", "huber")),
        huber_delta=float(getattr(cfg, "autoencoder_huber_delta", 1.0)),
        grad_clip=float(getattr(cfg, "autoencoder_grad_clip", 5.0)),
        training_window_days=getattr(cfg, "autoencoder_training_window_days", 1095),
        calibration_days=int(getattr(cfg, "autoencoder_calibration_days", 180)),
        holdout_days=int(getattr(cfg, "autoencoder_holdout_days", 180)),
        validation_fraction=float(getattr(cfg, "autoencoder_validation_fraction", 0.15)),
        min_train_windows=int(getattr(cfg, "autoencoder_min_train_windows", 120)),
        evt_alpha=float(getattr(cfg, "autoencoder_evt_alpha", 0.02)),
        evt_tail_quantile=float(getattr(cfg, "autoencoder_evt_tail_quantile", 0.85)),
        threshold_method=str(getattr(cfg, "autoencoder_threshold_method", "evt")),
        score_aggregation=str(getattr(cfg, "autoencoder_score_aggregation", "hybrid")),
        input_clip=float(getattr(cfg, "autoencoder_input_clip", 8.0)),
        weight_strength=float(getattr(cfg, "autoencoder_weight_strength", 0.75)),
        min_weight=float(getattr(cfg, "autoencoder_min_weight", 0.50)),
        known_event_min_weight=float(
            getattr(cfg, "autoencoder_known_event_min_weight", 0.85)
        ),
        seed=int(getattr(cfg, "autoencoder_seed", 42)),
        device=str(getattr(cfg, "autoencoder_device", "auto")),
        num_workers=int(getattr(cfg, "autoencoder_num_workers", 0)),
    )


def _cache_key(raw_df: pd.DataFrame, ae_cfg: AutoencoderV2Config) -> str:
    dates = pd.to_datetime(raw_df["DateKey"])
    payload = {
        "config": asdict(ae_cfg),
        "rows": int(len(raw_df)),
        "min_date": str(dates.min()),
        "max_date": str(dates.max()),
        "products": sorted(
            int(value) for value in pd.Series(raw_df["ProductId"]).dropna().unique()
        ),
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def build_cached_autoencoder_profile(
    raw_df: pd.DataFrame,
    cfg: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build or load a fold-local autoencoder profile.

    Cache identity includes the fold's maximum date and every model/action
    hyperparameter, so a later origin or different candidate cannot reuse a
    semantically stale profile.
    """
    ae_cfg = config_from_framework(cfg)
    cache_root = Path(getattr(cfg, "autoencoder_cache_dir", "outputs/anomaly_cache"))
    key = _cache_key(raw_df, ae_cfg)
    profile_path = cache_root / f"{key}.pkl"
    metadata_path = cache_root / f"{key}.json"
    if profile_path.exists() and metadata_path.exists():
        profile = pd.read_pickle(profile_path)
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata = dict(metadata)
        metadata["cache_hit"] = True
        metadata["cache_key"] = key
        return profile, metadata

    profile, metadata, model, preprocessor = fit_score_systemic_autoencoder_v2(
        raw_df, ae_cfg
    )
    del model, preprocessor
    gc.collect()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_profile = profile_path.with_suffix(f".tmp-{Path(profile_path).suffix.lstrip('.')}")
    profile.to_pickle(tmp_profile)
    tmp_profile.replace(profile_path)
    tmp_metadata = metadata_path.with_suffix(".tmp.json")
    with tmp_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    tmp_metadata.replace(metadata_path)
    metadata = dict(metadata)
    metadata["cache_hit"] = False
    metadata["cache_key"] = key
    return profile, metadata


def attach_autoencoder_origin_features(
    feature_df: pd.DataFrame,
    daily_profile: pd.DataFrame,
) -> pd.DataFrame:
    source = daily_profile.copy()
    if "autoencoder_split" not in source:
        source["autoencoder_split"] = "holdout"
    lookup = source[
        [
            "DateKey",
            "autoencoder_score",
            "autoencoder_percentile",
            "autoencoder_flag",
            "autoencoder_score_mean_7",
            "autoencoder_score_mean_28",
            "autoencoder_flag_rate_28",
            "days_since_autoencoder_anomaly",
            "autoencoder_split",
        ]
    ].copy()
    feature_columns = [
        "autoencoder_score",
        "autoencoder_percentile",
        "autoencoder_flag",
        "autoencoder_score_mean_7",
        "autoencoder_score_mean_28",
        "autoencoder_flag_rate_28",
        "days_since_autoencoder_anomaly",
    ]
    for column in feature_columns:
        lookup[column] = pd.to_numeric(lookup[column], errors="coerce").astype(float)
    # Expose only strictly post-calibration, out-of-sample reconstruction
    # state to the forecasting model. Earlier rows remain valid training rows
    # and are handled by the model's fitted missing-value preprocessing.
    lookup.loc[lookup["autoencoder_split"].ne("holdout"), feature_columns] = np.nan
    lookup = lookup.drop(columns="autoencoder_split").rename(
        columns={
            "autoencoder_score": "autoencoder_score_lag0",
            "autoencoder_percentile": "autoencoder_percentile_lag0",
            "autoencoder_flag": "autoencoder_flag_lag0",
        }
    )
    result = feature_df.merge(lookup, on="DateKey", how="left", validate="many_to_one")
    return result.sort_values(["ProductId", "DateKey"]).reset_index(drop=True)


def apply_autoencoder_weights_to_panel(
    panel: pd.DataFrame,
    daily_profile: pd.DataFrame,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    if "TargetDateKey" not in panel:
        raise ValueError("Panel must contain TargetDateKey")
    lookup = daily_profile[
        ["DateKey", "autoencoder_weight", "autoencoder_score", "autoencoder_flag"]
    ].rename(
        columns={
            "DateKey": "TargetDateKey",
            "autoencoder_weight": "autoencoder_weight_raw",
            "autoencoder_score": "target_autoencoder_score",
            "autoencoder_flag": "target_autoencoder_flag",
        }
    )
    result = panel.merge(lookup, on="TargetDateKey", how="left", validate="many_to_one")
    base = pd.to_numeric(
        result.get("sample_weight", pd.Series(1.0, index=result.index)), errors="coerce"
    ).fillna(1.0)
    ae_weight = pd.to_numeric(result["autoencoder_weight_raw"], errors="coerce").fillna(1.0)
    combined = base.to_numpy(dtype=float) * ae_weight.to_numpy(dtype=float)
    if normalize and combined.size:
        mean = float(np.mean(combined))
        if math.isfinite(mean) and mean > 0:
            combined /= mean
    result["sample_weight"] = combined
    return result
