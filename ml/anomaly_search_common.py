"""Shared candidate, metric and configuration utilities for overnight search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from framework import Config, compute_metrics
from pipeline import DEVELOPMENT_ORIGINS, recent_benchmark_origins
from systemic_autoencoder_v2 import AutoencoderV2Config


@dataclass(frozen=True)
class SearchProfile:
    name: str
    autoencoder_trials: int
    autoencoder_cutoffs: int
    autoencoder_seeds: tuple[int, ...]
    autoencoder_epoch_cap: int
    statistical_trials: int
    autoencoder_proxy_top: int
    proxy_development_origins: int
    proxy_benchmark_origins: int
    neural_top: int
    neural_development_origins: int
    neural_benchmark_origins: int
    neural_epochs: int
    neural_seeds: tuple[int, ...]
    confirmation_top: int
    confirmation_development_origins: int
    confirmation_benchmark_origins: int
    confirmation_epochs: int
    confirmation_seeds: tuple[int, ...]


PROFILES: dict[str, SearchProfile] = {
    "smoke": SearchProfile(
        name="smoke",
        autoencoder_trials=2,
        autoencoder_cutoffs=1,
        autoencoder_seeds=(42,),
        autoencoder_epoch_cap=3,
        statistical_trials=1,
        autoencoder_proxy_top=1,
        proxy_development_origins=1,
        proxy_benchmark_origins=1,
        neural_top=1,
        neural_development_origins=1,
        neural_benchmark_origins=1,
        neural_epochs=2,
        neural_seeds=(42,),
        confirmation_top=1,
        confirmation_development_origins=1,
        confirmation_benchmark_origins=1,
        confirmation_epochs=2,
        confirmation_seeds=(42,),
    ),
    "overnight": SearchProfile(
        name="overnight",
        autoencoder_trials=36,
        autoencoder_cutoffs=3,
        autoencoder_seeds=(42, 123),
        autoencoder_epoch_cap=180,
        statistical_trials=24,
        autoencoder_proxy_top=10,
        proxy_development_origins=6,
        proxy_benchmark_origins=8,
        neural_top=5,
        neural_development_origins=6,
        neural_benchmark_origins=8,
        neural_epochs=25,
        neural_seeds=(42,),
        confirmation_top=2,
        confirmation_development_origins=12,
        confirmation_benchmark_origins=12,
        confirmation_epochs=45,
        confirmation_seeds=(42, 123, 777),
    ),
    "weekend": SearchProfile(
        name="weekend",
        autoencoder_trials=72,
        autoencoder_cutoffs=4,
        autoencoder_seeds=(42, 123, 777),
        autoencoder_epoch_cap=240,
        statistical_trials=48,
        autoencoder_proxy_top=16,
        proxy_development_origins=9,
        proxy_benchmark_origins=16,
        neural_top=8,
        neural_development_origins=9,
        neural_benchmark_origins=12,
        neural_epochs=35,
        neural_seeds=(42, 123),
        confirmation_top=3,
        confirmation_development_origins=12,
        confirmation_benchmark_origins=24,
        confirmation_epochs=60,
        confirmation_seeds=(42, 123, 777),
    ),
    "exhaustive": SearchProfile(
        name="exhaustive",
        autoencoder_trials=120,
        autoencoder_cutoffs=5,
        autoencoder_seeds=(42, 123, 777),
        autoencoder_epoch_cap=320,
        statistical_trials=72,
        autoencoder_proxy_top=24,
        proxy_development_origins=12,
        proxy_benchmark_origins=24,
        neural_top=12,
        neural_development_origins=12,
        neural_benchmark_origins=24,
        neural_epochs=45,
        neural_seeds=(42, 123),
        confirmation_top=4,
        confirmation_development_origins=12,
        confirmation_benchmark_origins=52,
        confirmation_epochs=75,
        confirmation_seeds=(42, 123, 777, 2026),
    ),
}


CONFIG_FIELDS = set(Config.__dataclass_fields__)


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_id(payload: dict[str, Any], prefix: str) -> str:
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def candidate(
    family: str,
    name: str,
    config: dict[str, Any],
    *,
    diagnostic: dict[str, Any] | None = None,
    parents: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "family": family,
        "name": name,
        "config": config,
        "diagnostic": diagnostic,
        "parents": parents or [],
    }
    payload["id"] = stable_id(payload, family[:4])
    return payload


def control_candidate() -> dict[str, Any]:
    return candidate("control", "control", {"anomaly_mode": "off"})


def _ae_base_configs() -> list[dict[str, Any]]:
    return [
        {
            "window": 28,
            "representation": "weekday_residual",
            "architecture": "conv",
            "hidden_dim": 128,
            "latent_dim": 16,
            "dropout": 0.10,
            "max_epochs": 180,
            "patience": 24,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "noise_std": 0.03,
            "loss": "huber",
            "training_window_days": 1095,
            "calibration_days": 180,
            "holdout_days": 180,
            "evt_alpha": 0.02,
            "evt_tail_quantile": 0.85,
            "threshold_method": "evt",
            "score_aggregation": "hybrid",
            "weight_strength": 0.75,
            "min_weight": 0.50,
            "known_event_min_weight": 0.85,
        },
        {
            "window": 56,
            "representation": "weekday_residual",
            "architecture": "mlp",
            "hidden_dim": 256,
            "latent_dim": 32,
            "dropout": 0.15,
            "max_epochs": 180,
            "patience": 24,
            "batch_size": 64,
            "learning_rate": 7.5e-4,
            "weight_decay": 1e-4,
            "noise_std": 0.02,
            "loss": "huber",
            "training_window_days": 730,
            "calibration_days": 180,
            "holdout_days": 180,
            "evt_alpha": 0.02,
            "evt_tail_quantile": 0.90,
            "threshold_method": "empirical",
            "score_aggregation": "mean",
            "weight_strength": 0.50,
            "min_weight": 0.60,
            "known_event_min_weight": 0.90,
        },
        {
            "window": 28,
            "representation": "level_residual_availability",
            "architecture": "gru",
            "hidden_dim": 128,
            "latent_dim": 24,
            "dropout": 0.10,
            "max_epochs": 160,
            "patience": 20,
            "batch_size": 64,
            "learning_rate": 7.5e-4,
            "weight_decay": 1e-5,
            "noise_std": 0.01,
            "loss": "mse",
            "training_window_days": 1095,
            "calibration_days": 180,
            "holdout_days": 180,
            "evt_alpha": 0.01,
            "evt_tail_quantile": 0.85,
            "threshold_method": "empirical",
            "score_aggregation": "hybrid",
            "weight_strength": 0.40,
            "min_weight": 0.70,
            "known_event_min_weight": 0.90,
        },
    ]


def generate_autoencoder_candidates(
    count: int,
    *,
    seed: int,
    epoch_cap: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    raw: list[dict[str, Any]] = _ae_base_configs()
    choices: dict[str, list[Any]] = {
        "window": [14, 28, 42, 56, 84],
        "representation": [
            "log_level",
            "weekday_residual",
            "level_residual",
            "residual_availability",
            "level_residual_availability",
        ],
        "architecture": ["mlp", "conv", "gru"],
        "hidden_dim": [64, 128, 256],
        "latent_dim": [8, 16, 24, 32, 48],
        "dropout": [0.0, 0.05, 0.10, 0.20],
        "max_epochs": [100, 140, 180, 240, 320],
        "patience": [15, 24, 36],
        "batch_size": [32, 64, 128, 256],
        "learning_rate": [3e-4, 5e-4, 7.5e-4, 1e-3, 2e-3],
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
        "noise_std": [0.0, 0.01, 0.03, 0.05, 0.08],
        "loss": ["mse", "huber"],
        "training_window_days": [365, 730, 1095, 1460, None],
        "calibration_days": [90, 180, 270],
        "holdout_days": [90, 180, 270],
        "evt_alpha": [0.005, 0.01, 0.02, 0.05],
        "evt_tail_quantile": [0.80, 0.85, 0.90, 0.95],
        "threshold_method": ["evt", "empirical"],
        "score_aggregation": ["mean", "p95", "hybrid"],
        "weight_strength": [0.20, 0.40, 0.75, 1.0, 1.5],
        "min_weight": [0.30, 0.50, 0.70, 0.85],
        "known_event_min_weight": [0.80, 0.90, 1.0],
    }
    while len(raw) < count * 2:
        config = {key: _python_value(rng.choice(values)) for key, values in choices.items()}
        config["max_epochs"] = int(min(int(config["max_epochs"]), epoch_cap))
        config["window"] = int(config["window"])
        config["hidden_dim"] = int(config["hidden_dim"])
        config["latent_dim"] = int(config["latent_dim"])
        config["patience"] = int(min(int(config["patience"]), max(2, config["max_epochs"] // 3)))
        config["batch_size"] = int(config["batch_size"])
        config["calibration_days"] = int(config["calibration_days"])
        config["holdout_days"] = int(config["holdout_days"])
        if config["training_window_days"] is not None:
            config["training_window_days"] = int(config["training_window_days"])
        raw.append(config)

    unique: dict[str, dict[str, Any]] = {}
    for config in raw:
        config = dict(config)
        config["max_epochs"] = min(int(config["max_epochs"]), epoch_cap)
        config["patience"] = min(
            int(config["patience"]), max(1, int(config["max_epochs"]))
        )
        key = stable_id(config, "ae")
        unique.setdefault(key, config)
        if len(unique) >= count:
            break

    candidates = []
    for index, config in enumerate(unique.values(), start=1):
        framework_config = {
            "anomaly_mode": "features",
            "anomaly_source": "autoencoder",
            **{f"autoencoder_{key}": value for key, value in config.items()},
        }
        # Dataclass uses max_epochs rather than epochs; all generated keys map.
        candidates.append(
            candidate(
                "autoencoder",
                f"ae_{index:03d}_{config['architecture']}_{config['representation']}_w{config['window']}",
                framework_config,
                diagnostic=config,
            )
        )
    return candidates


def generate_statistical_candidates(count: int, *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    defaults = [
        {
            "anomaly_mode": "features",
            "anomaly_source": "statistical",
            "anomaly_rolling_window": 180,
            "anomaly_min_history": 28,
            "anomaly_scale_floor": 0.10,
            "anomaly_evt_alpha": 0.01,
            "anomaly_evt_tail_quantile": 0.90,
            "anomaly_weight_strength": 1.0,
            "anomaly_min_weight": 0.20,
            "anomaly_known_event_min_weight": 0.65,
            "anomaly_systemic_min_weight": 0.50,
        },
        {
            "anomaly_mode": "both",
            "anomaly_source": "statistical",
            "anomaly_rolling_window": 365,
            "anomaly_min_history": 56,
            "anomaly_scale_floor": 0.15,
            "anomaly_evt_alpha": 0.02,
            "anomaly_evt_tail_quantile": 0.85,
            "anomaly_weight_strength": 0.35,
            "anomaly_min_weight": 0.70,
            "anomaly_known_event_min_weight": 0.95,
            "anomaly_systemic_min_weight": 0.85,
        },
    ]
    choices: dict[str, list[Any]] = {
        "anomaly_mode": ["features", "weight", "both"],
        "anomaly_rolling_window": [60, 90, 180, 270, 365, 540],
        "anomaly_min_history": [21, 28, 42, 56, 84],
        "anomaly_scale_floor": [0.05, 0.10, 0.15, 0.25, 0.40],
        "anomaly_evt_alpha": [0.0025, 0.005, 0.01, 0.02, 0.05],
        "anomaly_evt_tail_quantile": [0.80, 0.85, 0.90, 0.95],
        "anomaly_weight_strength": [0.15, 0.30, 0.50, 0.75, 1.0, 1.5],
        "anomaly_min_weight": [0.20, 0.40, 0.60, 0.75, 0.90],
        "anomaly_known_event_min_weight": [0.65, 0.80, 0.90, 1.0],
        "anomaly_systemic_min_weight": [0.50, 0.70, 0.85, 1.0],
    }
    raw = list(defaults)
    while len(raw) < count * 2:
        config = {key: _python_value(rng.choice(values)) for key, values in choices.items()}
        config["anomaly_source"] = "statistical"
        for key in ("anomaly_rolling_window", "anomaly_min_history"):
            config[key] = int(config[key])
        config["anomaly_min_history"] = min(
            config["anomaly_min_history"], config["anomaly_rolling_window"]
        )
        raw.append(config)
    unique: dict[str, dict[str, Any]] = {}
    for config in raw:
        config = dict(config)
        config["anomaly_min_history"] = min(
            int(config["anomaly_min_history"]), int(config["anomaly_rolling_window"])
        )
        unique.setdefault(stable_id(config, "stat"), config)
        if len(unique) >= count:
            break
    return [
        candidate(
            "statistical",
            f"stat_{index:03d}_{config['anomaly_mode']}_rw{config['anomaly_rolling_window']}",
            config,
        )
        for index, config in enumerate(unique.values(), start=1)
    ]


def autoencoder_action_variants(base: dict[str, Any]) -> list[dict[str, Any]]:
    variants = []
    for mode in ("features", "both"):
        clone = json.loads(json.dumps(base))
        clone["name"] = f"{base['name']}_{mode}"
        clone["config"]["anomaly_mode"] = mode
        clone.pop("id", None)
        clone["id"] = stable_id(clone, "aeac")
        variants.append(clone)
    return variants


def make_hybrid_candidate(
    statistical: dict[str, Any],
    autoencoder: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    config = dict(statistical["config"])
    config.update(autoencoder["config"])
    config["anomaly_mode"] = "both"
    config["anomaly_source"] = "hybrid"
    return candidate(
        "hybrid",
        f"hybrid_{index:02d}_{statistical['id']}_{autoencoder['id']}",
        config,
        diagnostic=autoencoder.get("diagnostic"),
        parents=[statistical["id"], autoencoder["id"]],
    )


def apply_candidate_config(cfg: Config, candidate_payload: dict[str, Any]) -> Config:
    for key, value in candidate_payload.get("config", {}).items():
        if key not in CONFIG_FIELDS:
            raise ValueError(f"Candidate contains unknown Config field: {key}")
        setattr(cfg, key, value)
    return cfg


def selected_forecasting_config() -> Config:
    """The confirmed forecasting estimator around which anomaly search runs."""
    cfg = Config()
    cfg.training_window_days = None
    cfg.recency_half_life_days = None
    cfg.baseline_variant = "weighted_4321"
    cfg.enable_trend_features = False
    cfg.c2_feature_groups = ("price", "campaign", "lifecycle", "market", "event")
    cfg.nn_loss = "mse"
    cfg.nn_target_mode = "residual"
    cfg.enable_channel_history_features = False
    cfg.channel_aux_weight = 0.0
    cfg.tree_target_mode = "log1p"
    cfg.xgboost_target_mode = "residual"
    cfg.lightgbm_target_mode = "log1p"
    cfg.structured_worker_timeout_seconds = 3600
    return cfg


def evenly_spaced_origins(origins: Iterable[pd.Timestamp], count: int) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(origins).sort_values()
    if count >= len(values):
        return values
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return pd.DatetimeIndex(values[np.unique(positions)])


def development_origins(count: int) -> pd.DatetimeIndex:
    return evenly_spaced_origins(DEVELOPMENT_ORIGINS, count)


def benchmark_origins(train: pd.DataFrame, count: int, cfg: Config) -> pd.DatetimeIndex:
    local = Config(**asdict(cfg))
    local.n_cv_folds = count
    return recent_benchmark_origins(train, local)


def prediction_column(model: str) -> str:
    mapping = {
        "DynamicRidge": "pred_DynamicRidge",
        "NeuralNet": "pred_NeuralNet",
        "LightGBM": "pred_LightGBM",
    }
    try:
        return mapping[model]
    except KeyError as exc:
        raise ValueError(f"Unsupported search model: {model}") from exc


def _valid_oof(oof: pd.DataFrame, prediction: str) -> pd.DataFrame:
    if oof.empty:
        return oof.copy()
    available = oof.get("ProductAvailable", pd.Series(True, index=oof.index))
    mask = available.astype("boolean").fillna(False).astype(bool)
    mask &= pd.to_numeric(oof["actual"], errors="coerce").notna()
    mask &= pd.to_numeric(oof[prediction], errors="coerce").notna()
    return oof.loc[mask].copy()


def metric_row(frame: pd.DataFrame, prediction: str) -> dict[str, Any]:
    valid = _valid_oof(frame, prediction)
    if valid.empty:
        return {"WAPE": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "BiasRatio": float("nan"), "n": 0}
    metrics = compute_metrics(
        valid["actual"].to_numpy(dtype=float),
        valid[prediction].to_numpy(dtype=float),
    )
    return {
        key: metrics[key]
        for key in ("WAPE", "MAE", "RMSE", "BiasRatio", "n")
    }


def summarize_oof(oof: pd.DataFrame, model: str) -> dict[str, Any]:
    pred = prediction_column(model)
    valid = _valid_oof(oof, pred)
    summary: dict[str, Any] = {"global": metric_row(valid, pred)}
    summary["by_origin"] = {
        str(pd.Timestamp(origin).date()): metric_row(frame, pred)
        for origin, frame in valid.groupby("origin", sort=True)
    }
    summary["by_horizon"] = {
        str(int(horizon)): metric_row(frame, pred)
        for horizon, frame in valid.groupby("horizon", sort=True)
    }
    if "validation_stratum" in valid:
        summary["by_stratum"] = {
            str(stratum): metric_row(frame, pred)
            for stratum, frame in valid.groupby("validation_stratum", sort=True)
        }
    else:
        summary["by_stratum"] = {}
    if not valid.empty:
        cutoff = float(valid["actual"].quantile(0.90))
        top = valid[valid["actual"] >= cutoff]
        summary["top_actual_decile"] = {
            "cutoff": cutoff,
            **metric_row(top, pred),
        }
    else:
        summary["top_actual_decile"] = {"cutoff": float("nan"), **metric_row(valid, pred)}
    return summary


def bootstrap_origin_improvement(
    control_oof: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    model: str,
    *,
    samples: int = 4000,
    seed: int = 42,
) -> dict[str, float]:
    pred = prediction_column(model)
    control = _valid_oof(control_oof, pred)
    challenger = _valid_oof(candidate_oof, pred)
    keys = ["ProductId", "DateKey", "origin"]
    merged = control[keys + ["actual", pred]].merge(
        challenger[keys + [pred]],
        on=keys,
        suffixes=("_control", "_candidate"),
        validate="one_to_one",
    )
    if merged.empty:
        return {
            "mean_relative_improvement": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_improvement_positive": float("nan"),
        }
    origins = np.asarray(sorted(merged["origin"].unique()))
    rng = np.random.default_rng(seed)
    improvements = []
    for _ in range(samples):
        sampled = rng.choice(origins, size=len(origins), replace=True)
        frames = [merged[merged["origin"] == origin] for origin in sampled]
        boot = pd.concat(frames, ignore_index=True)
        denominator = float(np.abs(boot["actual"]).sum())
        if denominator <= 0:
            continue
        control_wape = float(
            np.abs(boot["actual"] - boot[f"{pred}_control"]).sum() / denominator
        )
        candidate_wape = float(
            np.abs(boot["actual"] - boot[f"{pred}_candidate"]).sum() / denominator
        )
        improvements.append((control_wape - candidate_wape) / control_wape)
    values = np.asarray(improvements, dtype=float)
    return {
        "mean_relative_improvement": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_improvement_positive": float(np.mean(values > 0.0)),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    tmp.replace(path)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
