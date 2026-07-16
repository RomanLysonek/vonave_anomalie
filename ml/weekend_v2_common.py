"""Shared utilities for the evidence-driven weekend-v2 forecast search.

The first overnight experiment showed that anomaly specialists can be useful
without being standalone winners.  This module therefore treats anomaly models,
recent-regime models and the canonical network as experts in a mixture rather
than forcing a binary anomaly-mode decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np
import pandas as pd

from anomaly_search_common import (
    apply_candidate_config,
    autoencoder_action_variants,
    candidate,
    control_candidate,
    load_json,
    stable_id,
    write_json,
)
from framework import ANOMALY_ORIGIN_FEATURES, AUTOENCODER_ORIGIN_FEATURES, compute_metrics


@dataclass(frozen=True)
class WeekendV2Profile:
    name: str
    statistical_candidates: int
    regime_candidates: int
    autoencoder_top: int
    screen_development_origins: int
    screen_benchmark_origins: int
    screen_epochs: int
    screen_seeds: tuple[int, ...]
    refine_top: int
    refine_development_origins: int
    refine_benchmark_origins: int
    refine_epochs: int
    refine_seeds: tuple[int, ...]
    confirmation_top: int
    confirmation_development_origins: int
    confirmation_benchmark_origins: int
    confirmation_epochs: int
    confirmation_seeds: tuple[int, ...]
    ensemble_samples: int


WEEKEND_V2_PROFILES: dict[str, WeekendV2Profile] = {
    "smoke": WeekendV2Profile(
        "smoke", 3, 2, 1, 2, 2, 2, (42,), 2, 2, 2, 3, (42,),
        1, 2, 2, 3, (42,), 500,
    ),
    "weekend-v2": WeekendV2Profile(
        "weekend-v2", 24, 10, 3, 5, 8, 10, (42,), 10, 10, 18, 30,
        (42, 123), 5, 12, 36, 55, (42, 123, 777), 60000,
    ),
    "exhaustive-v2": WeekendV2Profile(
        "exhaustive-v2", 44, 20, 8, 12, 24, 22, (42, 123), 18, 12, 36,
        45, (42, 123), 8, 12, 52, 80, (42, 123, 777, 2026), 150000,
    ),
}


def _dedupe(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        result.setdefault(item["id"], item)
    return list(result.values())


def _load_prior_best_stat(prior_root: Path) -> dict[str, Any] | None:
    recommendation = prior_root / "recommendation.json"
    if recommendation.exists():
        payload = load_json(recommendation)
        comparisons = payload.get("comparisons", [])
        statistical = [
            row for row in comparisons
            if row.get("candidate", {}).get("family") == "statistical"
        ]
        if statistical:
            statistical.sort(
                key=lambda row: (
                    row.get("development_relative_improvement", -999.0),
                    -row.get("benchmark_relative_change", 999.0),
                ),
                reverse=True,
            )
            return statistical[0]["candidate"]
    candidate_path = prior_root / "candidates" / "stat-322e3ec6f7ee.json"
    return load_json(candidate_path) if candidate_path.exists() else None


def generate_statistical_neighborhood(
    count: int, *, seed: int, prior_root: Path
) -> list[dict[str, Any]]:
    """Generate a local but policy-diverse search around the overnight near-winner."""
    prior = _load_prior_best_stat(prior_root)
    base = dict((prior or {}).get("config", {}))
    if not base:
        base = {
            "anomaly_mode": "both",
            "anomaly_source": "statistical",
            "anomaly_rolling_window": 90,
            "anomaly_min_history": 21,
            "anomaly_scale_floor": 0.40,
            "anomaly_evt_alpha": 0.05,
            "anomaly_evt_tail_quantile": 0.95,
            "anomaly_weight_strength": 0.30,
            "anomaly_min_weight": 0.75,
            "anomaly_known_event_min_weight": 0.90,
            "anomaly_systemic_min_weight": 0.70,
        }
    base.setdefault("anomaly_weight_policy", "downweight")
    base.setdefault("anomaly_max_weight", 2.0)

    # Hand-authored hypotheses come first so smoke profiles test meaningful
    # alternatives rather than arbitrary random combinations.
    hypotheses: list[dict[str, Any]] = []
    for mode, policy, strength, floor in [
        ("both", "downweight", 0.15, 0.85),
        ("both", "downweight", 0.30, 0.75),
        ("features", "downweight", 0.0, 1.0),
        ("weight", "negative_only", 0.30, 0.75),
        ("both", "negative_only", 0.30, 0.75),
        ("weight", "hard_example", 0.12, 1.0),
        ("both", "hard_example", 0.12, 1.0),
        ("weight", "signed", 0.12, 0.85),
        ("both", "signed", 0.12, 0.85),
    ]:
        cfg = dict(base)
        cfg.update({
            "anomaly_mode": mode,
            "anomaly_weight_policy": policy,
            "anomaly_weight_strength": strength,
            "anomaly_min_weight": floor,
            "anomaly_max_weight": 1.50 if policy in {"hard_example", "signed"} else 2.0,
        })
        hypotheses.append(cfg)

    rng = np.random.default_rng(seed)
    choices = {
        "anomaly_mode": ["features", "weight", "both"],
        "anomaly_rolling_window": [45, 60, 90, 120, 180, 270],
        "anomaly_min_history": [14, 21, 28, 42, 56],
        "anomaly_scale_floor": [0.15, 0.25, 0.40, 0.60],
        "anomaly_evt_alpha": [0.01, 0.02, 0.05, 0.10],
        "anomaly_evt_tail_quantile": [0.85, 0.90, 0.95],
        "anomaly_weight_policy": [
            "downweight", "negative_only", "hard_example", "signed"
        ],
        "anomaly_weight_strength": [0.08, 0.12, 0.20, 0.30, 0.45],
        "anomaly_min_weight": [0.65, 0.75, 0.85, 0.95],
        "anomaly_max_weight": [1.20, 1.35, 1.50, 1.75],
        "anomaly_known_event_min_weight": [0.90, 1.0],
        "anomaly_systemic_min_weight": [0.70, 0.85, 1.0],
    }
    while len(hypotheses) < count * 3:
        cfg = dict(base)
        for key, values in choices.items():
            value = rng.choice(values)
            cfg[key] = value.item() if isinstance(value, np.generic) else value
        cfg["anomaly_source"] = "statistical"
        cfg["anomaly_rolling_window"] = int(cfg["anomaly_rolling_window"])
        cfg["anomaly_min_history"] = min(
            int(cfg["anomaly_min_history"]), cfg["anomaly_rolling_window"]
        )
        if cfg["anomaly_mode"] == "features":
            cfg["anomaly_weight_strength"] = 0.0
        hypotheses.append(cfg)

    unique: dict[str, dict[str, Any]] = {}
    for cfg in hypotheses:
        cfg = dict(cfg)
        cfg["anomaly_min_history"] = min(
            int(cfg["anomaly_min_history"]), int(cfg["anomaly_rolling_window"])
        )
        key = stable_id(cfg, "statv2")
        unique.setdefault(key, cfg)
        if len(unique) >= count:
            break
    output = []
    for index, cfg in enumerate(unique.values(), 1):
        output.append(candidate(
            "statistical",
            f"v2_stat_{index:02d}_{cfg['anomaly_weight_policy']}_rw{cfg['anomaly_rolling_window']}",
            cfg,
        ))
    return output


def generate_regime_specialists(count: int) -> list[dict[str, Any]]:
    """Time-diversified neural experts; most deliberately have anomaly_mode off."""
    raw = [
        (365, None, "mse", "residual"),
        (540, None, "mse", "residual"),
        (730, None, "mse", "residual"),
        (1095, None, "mse", "residual"),
        (None, 90.0, "mse", "residual"),
        (None, 180.0, "mse", "residual"),
        (None, 365.0, "mse", "residual"),
        (730, 180.0, "huber", "residual"),
        (1095, 365.0, "combined", "residual"),
        (730, 365.0, "logcosh", "residual"),
        (1095, 365.0, "mse", "log1p"),
        (540, 180.0, "combined", "log1p"),
        (None, 730.0, "huber", "residual"),
        (1460, 730.0, "mse", "residual"),
    ]
    output = []
    for index, (window, half_life, loss, target) in enumerate(raw[:count], 1):
        cfg = {
            "anomaly_mode": "off",
            "training_window_days": window,
            "recency_half_life_days": half_life,
            "nn_loss": loss,
            "nn_target_mode": target,
        }
        if loss == "combined":
            cfg["nn_combined_mse_weight"] = 0.50
        output.append(candidate(
            "regime",
            f"v2_regime_{index:02d}_w{window or 'all'}_hl{half_life or 'none'}_{loss}_{target}",
            cfg,
        ))
    return output


def load_top_autoencoder_specialists(
    prior_root: Path, count: int
) -> list[dict[str, Any]]:
    leaderboard = prior_root / "diagnostic_leaderboard.csv"
    if not leaderboard.exists() or count <= 0:
        return []
    rows = pd.read_csv(leaderboard).head(count)
    output: list[dict[str, Any]] = []
    for candidate_id in rows["candidate_id"].astype(str):
        path = prior_root / "candidates" / f"{candidate_id}.json"
        if not path.exists():
            continue
        base = load_json(path)
        variants = autoencoder_action_variants(base)
        # Features-only is the safer interpretation of the strong risk signal;
        # combined mode is retained as a deliberately different specialist.
        output.extend(variants)
    return output


def generate_weekend_v2_candidates(
    profile: WeekendV2Profile, *, seed: int, prior_root: Path
) -> list[dict[str, Any]]:
    stat = generate_statistical_neighborhood(
        profile.statistical_candidates, seed=seed, prior_root=prior_root
    )
    regime = generate_regime_specialists(profile.regime_candidates)
    ae = load_top_autoencoder_specialists(prior_root, profile.autoencoder_top)
    hybrids: list[dict[str, Any]] = []
    best_stat = stat[:2]
    for s in best_stat:
        for a in ae[:2]:
            cfg = dict(s["config"])
            cfg.update(a["config"])
            cfg["anomaly_mode"] = "both"
            cfg["anomaly_source"] = "hybrid"
            hybrids.append(candidate(
                "hybrid",
                f"v2_hybrid_{s['id']}_{a['id']}",
                cfg,
                parents=[s["id"], a["id"]],
                diagnostic=a.get("diagnostic"),
            ))
    return _dedupe([control_candidate(), *stat, *regime, *ae, *hybrids])


def valid_oof(frame: pd.DataFrame, prediction_columns: Iterable[str]) -> pd.DataFrame:
    columns = list(prediction_columns)
    available = frame.get("ProductAvailable", pd.Series(True, index=frame.index))
    mask = available.astype("boolean").fillna(False).astype(bool)
    mask &= pd.to_numeric(frame["actual"], errors="coerce").notna()
    for column in columns:
        mask &= pd.to_numeric(frame[column], errors="coerce").notna()
    return frame.loc[mask].copy()


def wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denominator = float(np.abs(actual).sum())
    return float(np.abs(actual - prediction).sum() / denominator) if denominator > 0 else float("nan")


def bias_ratio(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    return float((prediction - actual).sum() / denominator) if denominator > 0 else float("nan")


def forecast_result_score(payload: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    dev = float(payload["development"]["global"]["WAPE"])
    bench = float(payload["benchmark"]["global"]["WAPE"])
    control_dev = float(control["development"]["global"]["WAPE"])
    control_bench = float(control["benchmark"]["global"]["WAPE"])
    dev_imp = (control_dev - dev) / control_dev
    bench_imp = (control_bench - bench) / control_bench
    origin_control = control["development"].get("by_origin", {})
    origin_candidate = payload["development"].get("by_origin", {})
    improvements = []
    for origin in sorted(set(origin_control) & set(origin_candidate)):
        c = float(origin_control[origin]["WAPE"])
        x = float(origin_candidate[origin]["WAPE"])
        if np.isfinite(c) and c > 0 and np.isfinite(x):
            improvements.append((c - x) / c)
    stability = float(np.std(improvements)) if improvements else 1.0
    positive_share = float(np.mean(np.asarray(improvements) > 0)) if improvements else 0.0
    top_control = float(control["development"]["top_actual_decile"]["WAPE"])
    top_candidate = float(payload["development"]["top_actual_decile"]["WAPE"])
    top_imp = (top_control - top_candidate) / top_control
    score = (
        dev_imp
        + 0.35 * bench_imp
        + 0.10 * top_imp
        + 0.05 * positive_share
        - 0.12 * stability
        - 2.0 * max(-0.02 - bench_imp, 0.0)
    )
    return {
        "candidate_id": payload["candidate"]["id"],
        "name": payload["candidate"]["name"],
        "family": payload["candidate"]["family"],
        "development_WAPE": dev,
        "benchmark_WAPE": bench,
        "development_relative_improvement": dev_imp,
        "benchmark_relative_improvement": bench_imp,
        "top_decile_relative_improvement": top_imp,
        "origin_improvement_std": stability,
        "origin_positive_share": positive_share,
        "robust_score": score,
    }


def load_stage_results(stage_dir: Path) -> list[dict[str, Any]]:
    output = []
    if not stage_dir.exists():
        return output
    for path in sorted(stage_dir.glob("*/result.json")):
        payload = load_json(path)
        if payload.get("status") == "complete":
            payload["_result_path"] = str(path)
            output.append(payload)
    return output


def rank_forecast_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control = next(x for x in results if x["candidate"]["family"] == "control")
    rows = [forecast_result_score(x, control) for x in results]
    return sorted(rows, key=lambda row: row["robust_score"], reverse=True)


def merge_oof(results: list[dict[str, Any]], split: str) -> tuple[pd.DataFrame, list[str]]:
    """Merge member predictions and candidate-specific origin-safe state.

    Anomaly state is prefixed by candidate id.  Otherwise a control frame with
    anomaly mode disabled would silently erase the very gate features that the
    specialist models generated.
    """
    keys = ["ProductId", "horizon", "DateKey", "origin"]
    control = next(x for x in results if x["candidate"]["family"] == "control")
    control_path = Path(control["_result_path"]).parent / f"{split}_oof.parquet"
    frame = pd.read_parquet(control_path)
    member_columns: list[str] = []
    base_columns = [
        *keys, "origin_type", "strategy", "validation_stratum", "actual",
        "ProductAvailable", "baseline", "pred_SeasonalNaive", "pred_MovingAvg28",
    ]
    frame = frame[[column for column in dict.fromkeys(base_columns) if column in frame]].copy()
    origin_features = set(ANOMALY_ORIGIN_FEATURES) | set(AUTOENCODER_ORIGIN_FEATURES)
    for result in results:
        candidate_id = result["candidate"]["id"]
        path = Path(result["_result_path"]).parent / f"{split}_oof.parquet"
        member = pd.read_parquet(path)
        column = f"member__{candidate_id}"
        member_columns.append(column)
        available_features = [name for name in origin_features if name in member.columns]
        selected = member[keys + ["pred_NeuralNet", *available_features]].copy()
        rename = {"pred_NeuralNet": column}
        rename.update({
            name: f"feature__{candidate_id}__{name}" for name in available_features
        })
        selected = selected.rename(columns=rename)
        frame = frame.merge(selected, on=keys, how="inner", validate="one_to_one")
    frame = valid_oof(frame, member_columns).reset_index(drop=True)
    return frame, member_columns


def _prediction(frame: pd.DataFrame, members: list[str], weights: np.ndarray) -> np.ndarray:
    matrix = frame[members].to_numpy(dtype=float)
    return np.clip(matrix @ np.asarray(weights, dtype=float), 0.0, None)


def _origin_improvements(
    frame: pd.DataFrame, prediction: np.ndarray, control_column: str
) -> np.ndarray:
    work = frame.copy()
    work["_candidate"] = prediction
    values = []
    for _, group in work.groupby("origin", sort=True):
        c = wape(group["actual"].to_numpy(), group[control_column].to_numpy())
        x = wape(group["actual"].to_numpy(), group["_candidate"].to_numpy())
        if np.isfinite(c) and c > 0:
            values.append((c - x) / c)
    return np.asarray(values, dtype=float)


def search_convex_weights(
    frame: pd.DataFrame,
    members: list[str],
    *,
    samples: int,
    seed: int,
    reference_column: str,
    stability_penalty: float = 0.06,
    bias_penalty: float = 0.03,
    batch_size: int = 768,
) -> dict[str, Any]:
    """Search non-negative sum-to-one weights with vectorized WAPE scoring.

    The first implementation evaluated each Dirichlet draw through a pandas
    groupby.  That made a 50k-weight search needlessly expensive.  This version
    batches the matrix products and computes origin stability from precomputed
    row indices, keeping the exact same objective.
    """
    if not members:
        raise ValueError("At least one member is required")
    frame = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n_members = len(members)
    fixed = [np.eye(n_members, dtype=float)[i] for i in range(n_members)]
    fixed.append(np.full(n_members, 1.0 / n_members, dtype=float))
    if n_members == 2:
        fixed.extend(
            np.asarray([1.0 - alpha, alpha], dtype=float)
            for alpha in np.linspace(0.0, 1.0, 101)
        )
    draws = rng.dirichlet(
        np.ones(n_members, dtype=float) * 1.3, size=max(int(samples), 1)
    )
    weights_matrix = np.vstack([*fixed, draws]).astype(float, copy=False)

    matrix = frame[members].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=float)
    reference = frame[reference_column].to_numpy(dtype=float)
    denominator = float(np.abs(actual).sum())
    if denominator <= 0:
        raise ValueError("Cannot optimize WAPE with a zero actual-demand denominator")
    reference_wape = wape(actual, reference)

    origin_indices: list[np.ndarray] = []
    origin_denominators: list[float] = []
    origin_reference_wapes: list[float] = []
    for _, index in frame.groupby("origin", sort=True).groups.items():
        row_index = np.asarray(list(index), dtype=int)
        origin_actual = actual[row_index]
        origin_denominator = float(np.abs(origin_actual).sum())
        if origin_denominator <= 0:
            continue
        origin_indices.append(row_index)
        origin_denominators.append(origin_denominator)
        origin_reference_wapes.append(
            float(np.abs(origin_actual - reference[row_index]).sum() / origin_denominator)
        )

    best_objective = float("inf")
    best_weights: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None
    for offset in range(0, len(weights_matrix), max(int(batch_size), 1)):
        batch = weights_matrix[offset : offset + max(int(batch_size), 1)]
        prediction = np.clip(matrix @ batch.T, 0.0, None)
        absolute_error = np.abs(actual[:, None] - prediction)
        batch_wape = absolute_error.sum(axis=0) / denominator
        batch_bias = (prediction - actual[:, None]).sum(axis=0) / denominator

        if origin_indices:
            origin_improvements = np.empty(
                (len(batch), len(origin_indices)), dtype=float
            )
            for column, (row_index, origin_denominator, reference_origin_wape) in enumerate(
                zip(origin_indices, origin_denominators, origin_reference_wapes)
            ):
                candidate_origin_wape = (
                    absolute_error[row_index].sum(axis=0) / origin_denominator
                )
                if reference_origin_wape > 0:
                    origin_improvements[:, column] = (
                        reference_origin_wape - candidate_origin_wape
                    ) / reference_origin_wape
                else:
                    origin_improvements[:, column] = 0.0
            stability = origin_improvements.std(axis=1)
            positive_share = (origin_improvements > 0).mean(axis=1)
        else:
            stability = np.ones(len(batch), dtype=float)
            positive_share = np.zeros(len(batch), dtype=float)

        objectives = (
            batch_wape
            + stability_penalty * stability
            + bias_penalty * np.abs(batch_bias)
        )
        local_index = int(np.argmin(objectives))
        local_objective = float(objectives[local_index])
        if local_objective < best_objective:
            best_objective = local_objective
            best_weights = batch[local_index].copy()
            best_metrics = {
                "WAPE": float(batch_wape[local_index]),
                "BiasRatio": float(batch_bias[local_index]),
                "relative_improvement": float(
                    (reference_wape - batch_wape[local_index]) / reference_wape
                ),
                "origin_positive_share": float(positive_share[local_index]),
                "origin_improvement_std": float(stability[local_index]),
            }

    assert best_weights is not None and best_metrics is not None
    return {
        "weights": {
            member: float(value) for member, value in zip(members, best_weights)
        },
        "objective": best_objective,
        "metrics": best_metrics,
    }


def apply_weight_plan(
    frame: pd.DataFrame, plan: dict[str, Any], members: list[str]
) -> np.ndarray:
    work = frame.reset_index(drop=True)
    if plan["method"] in {"global_convex", "aggregate_reconciled"}:
        weights = np.asarray([plan["weights"].get(member, 0.0) for member in members])
        pred = _prediction(work, members, weights)
        if plan["method"] == "aggregate_reconciled":
            scales = {int(k): float(v) for k, v in plan.get("horizon_scales", {}).items()}
            pred = pred * work["horizon"].map(scales).fillna(1.0).to_numpy(dtype=float)
        return np.clip(pred, 0.0, None)
    if plan["method"] == "horizon_convex":
        output = np.empty(len(work), dtype=float)
        for horizon, index in work.groupby("horizon").groups.items():
            weight_map = plan["horizon_weights"].get(
                str(int(horizon)), plan["global_weights"]
            )
            weights = np.asarray([weight_map.get(member, 0.0) for member in members])
            row_index = np.asarray(list(index), dtype=int)
            output[row_index] = _prediction(work.iloc[row_index], members, weights)
        return np.clip(output, 0.0, None)
    if plan["method"] == "product_convex":
        output = np.empty(len(work), dtype=float)
        for product_id, index in work.groupby("ProductId").groups.items():
            weight_map = plan["product_weights"].get(
                str(int(product_id)), plan["global_weights"]
            )
            weights = np.asarray([weight_map.get(member, 0.0) for member in members])
            row_index = np.asarray(list(index), dtype=int)
            output[row_index] = _prediction(work.iloc[row_index], members, weights)
        return np.clip(output, 0.0, None)
    raise ValueError(f"Unsupported weight plan method: {plan['method']}")


def fit_horizon_plan(
    frame: pd.DataFrame,
    members: list[str],
    *,
    samples: int,
    seed: int,
    reference_column: str,
    shrinkage_rows: float = 500.0,
) -> dict[str, Any]:
    global_fit = search_convex_weights(
        frame, members, samples=samples, seed=seed, reference_column=reference_column
    )
    global_weights = global_fit["weights"]
    horizon_weights: dict[str, dict[str, float]] = {}
    for horizon, group in frame.groupby("horizon", sort=True):
        local = search_convex_weights(
            group,
            members,
            samples=max(1000, samples // 10),
            seed=seed + int(horizon),
            reference_column=reference_column,
            stability_penalty=0.0,
            bias_penalty=0.01,
        )["weights"]
        fraction = len(group) / (len(group) + shrinkage_rows)
        blended = {
            member: fraction * local[member] + (1.0 - fraction) * global_weights[member]
            for member in members
        }
        total = sum(blended.values())
        horizon_weights[str(int(horizon))] = {
            key: value / total for key, value in blended.items()
        }
    return {
        "method": "horizon_convex",
        "global_weights": global_weights,
        "horizon_weights": horizon_weights,
        "shrinkage_rows": shrinkage_rows,
    }


def fit_product_plan(
    frame: pd.DataFrame,
    members: list[str],
    *,
    samples: int,
    seed: int,
    reference_column: str,
    shrinkage_rows: float = 180.0,
) -> dict[str, Any]:
    """Fit product-specific mixtures shrunk strongly toward a global mixture.

    This tests the hypothesis that anomaly specialists help only selected SKUs.
    The shrinkage prevents 30 small per-product optimizations from becoming an
    oracle lookup table.
    """
    global_fit = search_convex_weights(
        frame, members, samples=samples, seed=seed, reference_column=reference_column
    )
    global_weights = global_fit["weights"]
    product_weights: dict[str, dict[str, float]] = {}
    for product_id, group in frame.groupby("ProductId", sort=True):
        local = search_convex_weights(
            group,
            members,
            samples=max(750, samples // 12),
            seed=seed + int(product_id),
            reference_column=reference_column,
            stability_penalty=0.02,
            bias_penalty=0.02,
        )["weights"]
        fraction = min(0.75, len(group) / (len(group) + shrinkage_rows))
        blended = {
            member: fraction * local[member] + (1.0 - fraction) * global_weights[member]
            for member in members
        }
        total = sum(blended.values())
        product_weights[str(int(product_id))] = {
            member: value / total for member, value in blended.items()
        }
    return {
        "method": "product_convex",
        "global_weights": global_weights,
        "product_weights": product_weights,
        "shrinkage_rows": shrinkage_rows,
    }


def fit_aggregate_reconciliation(
    frame: pd.DataFrame,
    base_plan: dict[str, Any],
    members: list[str],
    *,
    shrinkage_demand: float = 5000.0,
) -> dict[str, Any]:
    pred = apply_weight_plan(frame, base_plan, members)
    work = frame.copy()
    work["_pred"] = pred
    scales = {}
    for horizon, group in work.groupby("horizon", sort=True):
        actual_total = float(group["actual"].sum())
        pred_total = float(group["_pred"].sum())
        raw = actual_total / pred_total if pred_total > 0 else 1.0
        fraction = abs(actual_total) / (abs(actual_total) + shrinkage_demand)
        scales[str(int(horizon))] = float(np.clip(1.0 + fraction * (raw - 1.0), 0.85, 1.15))
    return {
        "method": "aggregate_reconciled",
        "weights": base_plan["weights"],
        "horizon_scales": scales,
        "shrinkage_demand": shrinkage_demand,
    }


def crossfit_plan(
    frame: pd.DataFrame,
    members: list[str],
    *,
    method: str,
    samples: int,
    seed: int,
    reference_column: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.full(len(frame), np.nan, dtype=float)
    origins = sorted(frame["origin"].unique())
    for index, origin in enumerate(origins):
        train = frame[frame["origin"] != origin]
        valid_index = frame.index[frame["origin"] == origin]
        if method == "global_convex":
            fit = search_convex_weights(
                train, members, samples=max(1000, samples // max(len(origins), 1)),
                seed=seed + index, reference_column=reference_column,
            )
            plan = {"method": "global_convex", "weights": fit["weights"]}
        elif method == "horizon_convex":
            plan = fit_horizon_plan(
                train, members, samples=max(1000, samples // max(len(origins), 1)),
                seed=seed + index, reference_column=reference_column,
            )
        elif method == "product_convex":
            plan = fit_product_plan(
                train, members, samples=max(1000, samples // max(len(origins), 1)),
                seed=seed + index, reference_column=reference_column,
            )
        else:
            raise ValueError(method)
        output[valid_index] = apply_weight_plan(frame.loc[valid_index], plan, members)
    if method == "global_convex":
        full = {
            "method": "global_convex",
            "weights": search_convex_weights(
                frame, members, samples=samples, seed=seed,
                reference_column=reference_column,
            )["weights"],
        }
    elif method == "horizon_convex":
        full = fit_horizon_plan(
            frame, members, samples=samples, seed=seed,
            reference_column=reference_column,
        )
    elif method == "product_convex":
        full = fit_product_plan(
            frame, members, samples=samples, seed=seed,
            reference_column=reference_column,
        )
    else:
        raise ValueError(method)
    return output, full


def bootstrap_probability(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    reference_column: str,
    *,
    samples: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Origin-block bootstrap of relative WAPE improvement.

    Sampling origin-level sufficient statistics is exactly equivalent to
    concatenating sampled origin frames, but is orders of magnitude cheaper.
    """
    frame = frame.reset_index(drop=True)
    actual = frame["actual"].to_numpy(dtype=float)
    reference = frame[reference_column].to_numpy(dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    origin_rows = []
    for _, index in frame.groupby("origin", sort=True).groups.items():
        row_index = np.asarray(list(index), dtype=int)
        origin_rows.append((
            float(np.abs(actual[row_index]).sum()),
            float(np.abs(actual[row_index] - reference[row_index]).sum()),
            float(np.abs(actual[row_index] - prediction[row_index]).sum()),
        ))
    stats = np.asarray(origin_rows, dtype=float)
    if stats.size == 0:
        return {
            "mean_relative_improvement": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_improvement_positive": 0.0,
        }
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(stats), size=(max(int(samples), 1), len(stats)))
    sampled = stats[picks].sum(axis=1)
    denominator = sampled[:, 0]
    valid = denominator > 0
    reference_wape = np.divide(
        sampled[:, 1], denominator, out=np.full(len(sampled), np.nan), where=valid
    )
    candidate_wape = np.divide(
        sampled[:, 2], denominator, out=np.full(len(sampled), np.nan), where=valid
    )
    improvement = np.divide(
        reference_wape - candidate_wape,
        reference_wape,
        out=np.full(len(sampled), np.nan),
        where=np.isfinite(reference_wape) & (reference_wape > 0),
    )
    array = improvement[np.isfinite(improvement)]
    if not len(array):
        return {
            "mean_relative_improvement": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_improvement_positive": 0.0,
        }
    return {
        "mean_relative_improvement": float(np.mean(array)),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "probability_improvement_positive": float(np.mean(array > 0)),
    }


def evaluate_prediction(
    frame: pd.DataFrame, prediction: np.ndarray, reference_column: str
) -> dict[str, Any]:
    actual = frame["actual"].to_numpy(dtype=float)
    reference = frame[reference_column].to_numpy(dtype=float)
    result = {
        "WAPE": wape(actual, prediction),
        "BiasRatio": bias_ratio(actual, prediction),
    }
    ref_wape = wape(actual, reference)
    result["relative_improvement"] = (ref_wape - result["WAPE"]) / ref_wape
    cutoff = float(frame["actual"].quantile(0.90))
    top = frame["actual"] >= cutoff
    result["top_decile_WAPE"] = wape(actual[top], prediction[top])
    result["reference_top_decile_WAPE"] = wape(actual[top], reference[top])
    result["top_decile_relative_change"] = (
        result["top_decile_WAPE"] - result["reference_top_decile_WAPE"]
    ) / result["reference_top_decile_WAPE"]
    if "validation_stratum" in frame:
        result["by_stratum"] = {}
        for stratum, group in frame.assign(_prediction=prediction).groupby("validation_stratum"):
            result["by_stratum"][str(stratum)] = {
                "WAPE": wape(group["actual"].to_numpy(), group["_prediction"].to_numpy()),
                "reference_WAPE": wape(group["actual"].to_numpy(), group[reference_column].to_numpy()),
            }
    return result


def save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(value, handle)
    tmp.replace(path)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_meta_features(frame: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Forecast-time-safe features for a residual gate/stacker."""
    features = pd.DataFrame(index=frame.index)
    for member in members:
        features[member] = pd.to_numeric(frame[member], errors="coerce")
    for column in ("baseline", "pred_SeasonalNaive", "pred_MovingAvg28"):
        features[column] = pd.to_numeric(frame.get(column), errors="coerce")
    product = pd.to_numeric(frame["ProductId"], errors="coerce")
    horizon = pd.to_numeric(frame["horizon"], errors="coerce")
    features["ProductId"] = product
    features["horizon"] = horizon
    for value in sorted(int(x) for x in product.dropna().unique()):
        features[f"product_is_{value}"] = (product == value).astype(float)
    for value in sorted(int(x) for x in horizon.dropna().unique()):
        features[f"horizon_is_{value}"] = (horizon == value).astype(float)
    date = pd.to_datetime(frame["DateKey"])
    month = date.dt.month.astype(float)
    day_of_week = date.dt.dayofweek.astype(float)
    features["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    features["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    features["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    features["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    matrix = features[members]
    features["member_mean"] = matrix.mean(axis=1)
    features["member_std"] = matrix.std(axis=1).fillna(0.0)
    features["member_range"] = matrix.max(axis=1) - matrix.min(axis=1)
    reference = matrix.iloc[:, 0]
    for member in members[1:]:
        features[f"delta__{member}"] = matrix[member] - reference
        features[f"relative_delta__{member}"] = (
            matrix[member] - reference
        ) / (reference.abs() + 1.0)
    for column in frame.columns:
        if (
            column.startswith("anomaly_")
            or column.startswith("autoencoder_")
            or column.startswith("feature__")
        ):
            if column in {"anomaly_mode", "anomaly_source"}:
                continue
            features[column] = pd.to_numeric(frame[column], errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def _make_meta_estimator(kind: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "ridge_residual":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=100.0),
        )
    if kind == "risk_gate":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.04,
                max_iter=180,
                max_leaf_nodes=9,
                min_samples_leaf=35,
                l2_regularization=15.0,
                random_state=seed,
            ),
        )
    raise ValueError(kind)


def crossfit_meta_model(
    frame: pd.DataFrame,
    members: list[str],
    *,
    kind: str,
    seed: int,
    reference_column: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a residual meta-model with leave-one-origin-out predictions."""
    features = build_meta_features(frame, members)
    target = frame["actual"].to_numpy(dtype=float) - frame[reference_column].to_numpy(dtype=float)
    output = np.full(len(frame), np.nan, dtype=float)
    origins = sorted(frame["origin"].unique())
    for index, origin in enumerate(origins):
        train_mask = frame["origin"] != origin
        valid_mask = frame["origin"] == origin
        estimator = _make_meta_estimator(kind, seed + index)
        estimator.fit(features.loc[train_mask], target[train_mask])
        correction = estimator.predict(features.loc[valid_mask])
        output[np.flatnonzero(valid_mask.to_numpy())] = np.clip(
            frame.loc[valid_mask, reference_column].to_numpy(dtype=float) + correction,
            0.0,
            None,
        )
    final_estimator = _make_meta_estimator(kind, seed)
    final_estimator.fit(features, target)
    bundle = {
        "kind": kind,
        "members": members,
        "reference_column": reference_column,
        "feature_columns": list(features.columns),
        "estimator": final_estimator,
    }
    return output, bundle


def apply_meta_model(frame: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    members = list(bundle["members"])
    features = build_meta_features(frame, members)
    for column in bundle["feature_columns"]:
        if column not in features:
            features[column] = np.nan
    features = features[bundle["feature_columns"]]
    correction = bundle["estimator"].predict(features)
    reference = frame[bundle["reference_column"]].to_numpy(dtype=float)
    return np.clip(reference + correction, 0.0, None)

def _make_specialist_gate_estimator(seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=220,
            max_leaf_nodes=7,
            min_samples_leaf=40,
            l2_regularization=20.0,
            random_state=seed,
        ),
    )


def _advantage_to_alpha(
    predicted_advantage: np.ndarray, *, scale: float, max_alpha: float
) -> np.ndarray:
    scale = max(float(scale), 1e-6)
    return float(max_alpha) * np.clip(
        np.asarray(predicted_advantage, dtype=float) / scale, 0.0, 1.0
    )


def crossfit_specialist_gate(
    frame: pd.DataFrame,
    *,
    control_column: str,
    specialist_column: str,
    seed: int,
    max_alpha: float = 0.75,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Learn when a specialist should replace part of the control forecast.

    The regression target is the specialist's absolute-error advantage over the
    control.  Only positive predicted advantage opens the gate, which makes this
    substantially safer and more interpretable than an unconstrained stacker.
    """
    members = [control_column, specialist_column]
    features = build_meta_features(frame, members)
    actual = frame["actual"].to_numpy(dtype=float)
    control = frame[control_column].to_numpy(dtype=float)
    specialist = frame[specialist_column].to_numpy(dtype=float)
    advantage = np.abs(actual - control) - np.abs(actual - specialist)
    output = np.full(len(frame), np.nan, dtype=float)
    origins = sorted(frame["origin"].unique())
    fold_scales: list[float] = []
    for index, origin in enumerate(origins):
        train_mask = (frame["origin"] != origin).to_numpy(dtype=bool)
        valid_mask = (frame["origin"] == origin).to_numpy(dtype=bool)
        estimator = _make_specialist_gate_estimator(seed + index)
        estimator.fit(features.loc[train_mask], advantage[train_mask])
        predicted_advantage = estimator.predict(features.loc[valid_mask])
        positive = advantage[train_mask & (advantage > 0)]
        if len(positive):
            scale = float(np.quantile(positive, 0.75))
        else:
            scale = float(np.quantile(np.abs(advantage[train_mask]), 0.75))
        scale = max(scale, 1.0)
        fold_scales.append(scale)
        alpha = _advantage_to_alpha(
            predicted_advantage, scale=scale, max_alpha=max_alpha
        )
        output[np.flatnonzero(valid_mask)] = np.clip(
            control[valid_mask] + alpha * (specialist[valid_mask] - control[valid_mask]),
            0.0,
            None,
        )

    final_estimator = _make_specialist_gate_estimator(seed)
    final_estimator.fit(features, advantage)
    positive = advantage[advantage > 0]
    final_scale = (
        float(np.quantile(positive, 0.75))
        if len(positive)
        else float(np.quantile(np.abs(advantage), 0.75))
    )
    bundle = {
        "kind": "specialist_gate",
        "members": members,
        "control_column": control_column,
        "specialist_column": specialist_column,
        "feature_columns": list(features.columns),
        "estimator": final_estimator,
        "advantage_scale": max(final_scale, 1.0),
        "max_alpha": float(max_alpha),
        "crossfit_scale_mean": float(np.mean(fold_scales)) if fold_scales else None,
    }
    return output, bundle


def apply_specialist_gate(frame: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    members = list(bundle["members"])
    features = build_meta_features(frame, members)
    for column in bundle["feature_columns"]:
        if column not in features:
            features[column] = np.nan
    predicted_advantage = bundle["estimator"].predict(
        features[bundle["feature_columns"]]
    )
    alpha = _advantage_to_alpha(
        predicted_advantage,
        scale=float(bundle["advantage_scale"]),
        max_alpha=float(bundle["max_alpha"]),
    )
    control = frame[bundle["control_column"]].to_numpy(dtype=float)
    specialist = frame[bundle["specialist_column"]].to_numpy(dtype=float)
    return np.clip(control + alpha * (specialist - control), 0.0, None)
