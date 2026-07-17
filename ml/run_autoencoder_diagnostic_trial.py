"""Evaluate one autoencoder candidate across temporal cutoffs and seeds.

This worker is intentionally one-candidate-per-process.  MPS memory is returned
when the process exits, making a long search substantially more robust than a
single Python process that repeatedly constructs neural networks.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from anomaly_search_common import load_json, write_json
from framework import Config, load_raw
from systemic_autoencoder_v2 import AutoencoderV2Config, fit_score_systemic_autoencoder_v2
from artifact_provenance import (
    diagnostic_trial_fingerprint,
    environment_metadata,
    output_fingerprints,
    result_body_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutoffs", required=True, help="Comma-separated YYYY-MM-DD dates")
    parser.add_argument("--seeds", required=True, help="Comma-separated integer seeds")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--save-scores", action="store_true")
    return parser.parse_args()


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 8 or np.nanstd(left[valid]) <= 1e-12 or np.nanstd(right[valid]) <= 1e-12:
        return 0.0
    value = spearmanr(left[valid], right[valid]).statistic
    return float(value) if np.isfinite(value) else 0.0


def _full_daily_arrays(raw: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    work = raw.copy()
    work["DateKey"] = pd.to_datetime(work["DateKey"])
    if "Quantity" not in work:
        work["Quantity"] = (
            pd.to_numeric(work.get("QuantityApp", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(work.get("QuantityWeb", 0.0), errors="coerce").fillna(0.0)
        )
    dates = pd.date_range(work["DateKey"].min(), work["DateKey"].max(), freq="D")
    products = sorted(work["ProductId"].dropna().astype(int).unique())
    available = (
        work["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
    )
    qty = (
        work.assign(_qty=pd.to_numeric(work["Quantity"], errors="coerce").where(available))
        .pivot(index="DateKey", columns="ProductId", values="_qty")
        .reindex(index=dates, columns=products)
        .to_numpy(dtype=float)
    )
    avail = (
        work.assign(_avail=available.astype(float))
        .pivot(index="DateKey", columns="ProductId", values="_avail")
        .reindex(index=dates, columns=products)
        .fillna(0.0)
        .to_numpy(dtype=bool)
    )
    return pd.DatetimeIndex(dates), qty, avail


def _future_seven_day_difficulty(raw: pd.DataFrame, origin_dates: pd.Series) -> np.ndarray:
    dates, qty, available = _full_daily_arrays(raw)
    date_to_position = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    lag_days = np.asarray([7, 14, 21, 28], dtype=int)
    lag_weights = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=float)
    output = np.full(len(origin_dates), np.nan, dtype=float)
    for row_index, raw_date in enumerate(pd.to_datetime(origin_dates)):
        origin_position = date_to_position.get(pd.Timestamp(raw_date))
        if origin_position is None:
            continue
        absolute_error = 0.0
        actual_sum = 0.0
        observations = 0
        for horizon in range(1, 8):
            target_position = origin_position + horizon
            if target_position >= len(dates):
                continue
            lag_positions = target_position - lag_days
            if np.any(lag_positions < 0):
                continue
            lag_values = qty[lag_positions, :].T
            observed = np.isfinite(lag_values)
            numerator = np.nansum(lag_values * lag_weights[None, :], axis=1)
            denominator = np.sum(observed * lag_weights[None, :], axis=1)
            baseline = np.divide(
                numerator,
                denominator,
                out=np.full(qty.shape[1], np.nan, dtype=float),
                where=denominator > 0,
            )
            actual = qty[target_position]
            valid = available[target_position] & np.isfinite(actual) & np.isfinite(baseline)
            if not valid.any():
                continue
            absolute_error += float(np.abs(actual[valid] - baseline[valid]).sum())
            actual_sum += float(np.abs(actual[valid]).sum())
            observations += int(valid.sum())
        if observations > 0 and actual_sum > 0:
            output[row_index] = absolute_error / actual_sum
    return output


def _run_metrics(
    scores: pd.DataFrame,
    metadata: dict,
    full_raw: pd.DataFrame,
    alpha: float,
) -> dict:
    holdout = scores[scores["autoencoder_split"] == "holdout"].copy()
    calibration = scores[scores["autoencoder_split"] == "calibration"].copy()
    holdout_rate = float(holdout["autoencoder_flag"].mean()) if len(holdout) else float("nan")
    calibration_rate = float(calibration["autoencoder_flag"].mean()) if len(calibration) else float("nan")
    ordinal = holdout["DateKey"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    score = holdout["autoencoder_score"].to_numpy(dtype=float)
    drift = _safe_spearman(ordinal, score)
    persistence = (
        float(pd.Series(score).autocorr(lag=1))
        if len(score) > 3 and np.nanstd(score) > 1e-12
        else 0.0
    )
    if not np.isfinite(persistence):
        persistence = 0.0
    future_difficulty = _future_seven_day_difficulty(full_raw, holdout["DateKey"])
    difficulty_correlation = _safe_spearman(score, future_difficulty)
    valid = np.isfinite(score) & np.isfinite(future_difficulty)
    top_decile_lift = 1.0
    if valid.sum() >= 20:
        threshold = float(np.quantile(score[valid], 0.90))
        high = valid & (score >= threshold)
        denominator = float(np.mean(future_difficulty[valid]))
        if high.any() and denominator > 0:
            top_decile_lift = float(np.mean(future_difficulty[high]) / denominator)
    event = holdout["autoencoder_known_event"].astype(bool).to_numpy()
    flags = holdout["autoencoder_flag"].astype(bool).to_numpy()
    event_rate = float(np.mean(flags[event])) if event.any() else 0.0
    non_event_rate = float(np.mean(flags[~event])) if (~event).any() else 0.0
    event_enrichment = (event_rate + 1e-4) / (non_event_rate + 1e-4)
    return {
        "calibration_flag_rate": calibration_rate,
        "holdout_flag_rate": holdout_rate,
        "calibration_far_error": abs(calibration_rate - alpha),
        "holdout_far_error": abs(holdout_rate - alpha),
        "time_drift_spearman": drift,
        "lag1_score_autocorrelation": persistence,
        "future_seven_day_wape_spearman": difficulty_correlation,
        "future_error_top_score_decile_lift": top_decile_lift,
        "event_flag_rate": event_rate,
        "non_event_flag_rate": non_event_rate,
        "event_enrichment": event_enrichment,
        "best_epoch": metadata["best_epoch"],
        "epochs_ran": metadata["epochs_ran"],
        "best_validation_loss": metadata["best_validation_loss"],
        "n_holdout": int(len(holdout)),
    }


def _aggregate(run_metrics: list[dict], score_frames: list[pd.DataFrame]) -> dict:
    numeric_keys = sorted(
        key
        for key in run_metrics[0]
        if isinstance(run_metrics[0][key], (int, float)) and key not in {"seed"}
    )
    means = {
        key: float(np.nanmean([float(row[key]) for row in run_metrics]))
        for key in numeric_keys
    }
    stability_values: list[float] = []
    by_cutoff: dict[str, list[pd.DataFrame]] = {}
    for frame in score_frames:
        by_cutoff.setdefault(str(frame["diagnostic_cutoff"].iloc[0]), []).append(frame)
    for frames in by_cutoff.values():
        if len(frames) < 2:
            continue
        for left_index in range(len(frames)):
            for right_index in range(left_index + 1, len(frames)):
                merged = frames[left_index][["DateKey", "autoencoder_score"]].merge(
                    frames[right_index][["DateKey", "autoencoder_score"]],
                    on="DateKey",
                    suffixes=("_left", "_right"),
                )
                stability_values.append(
                    _safe_spearman(
                        merged["autoencoder_score_left"].to_numpy(dtype=float),
                        merged["autoencoder_score_right"].to_numpy(dtype=float),
                    )
                )
    seed_stability = float(np.mean(stability_values)) if stability_values else 1.0

    predictive = max(-0.25, means["future_seven_day_wape_spearman"])
    lift_component = float(np.tanh(max(0.0, means["future_error_top_score_decile_lift"] - 1.0)))
    calibration_penalty = 4.0 * (
        means["calibration_far_error"] + means["holdout_far_error"]
    )
    drift_penalty = 0.30 * abs(means["time_drift_spearman"])
    persistence_penalty = 0.20 * max(0.0, means["lag1_score_autocorrelation"] - 0.85)
    objective = (
        0.45 * predictive
        + 0.20 * lift_component
        + 0.25 * seed_stability
        - calibration_penalty
        - drift_penalty
        - persistence_penalty
    )
    return {
        "means": means,
        "seed_score_stability": seed_stability,
        "diagnostic_objective": float(objective),
        "objective_components": {
            "predictive": predictive,
            "top_decile_lift_component": lift_component,
            "seed_stability": seed_stability,
            "calibration_penalty": calibration_penalty,
            "drift_penalty": drift_penalty,
            "persistence_penalty": persistence_penalty,
        },
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    candidate = load_json(Path(args.candidate))
    diagnostic = candidate.get("diagnostic")
    if not isinstance(diagnostic, dict):
        raise ValueError("Autoencoder candidate is missing diagnostic configuration")
    cutoffs = [pd.Timestamp(token) for token in args.cutoffs.split(",") if token]
    seeds = [int(token) for token in args.seeds.split(",") if token]
    train, _ = load_raw(Config())
    fingerprint = diagnostic_trial_fingerprint(
        candidate=candidate,
        train_data=train,
        cutoffs=cutoffs,
        seeds=seeds,
        device=args.device,
        save_scores=args.save_scores,
    )

    run_metrics: list[dict] = []
    score_frames: list[pd.DataFrame] = []
    try:
        base = AutoencoderV2Config(**diagnostic)
        for cutoff in cutoffs:
            history = train[train["DateKey"] <= cutoff].copy()
            if history.empty:
                continue
            for seed in seeds:
                cfg = replace(base, seed=seed, device=args.device)
                scores, metadata, _, _ = fit_score_systemic_autoencoder_v2(history, cfg)
                metrics = _run_metrics(scores, metadata, train, cfg.evt_alpha)
                metrics.update({"cutoff": str(cutoff.date()), "seed": seed})
                run_metrics.append(metrics)
                scores = scores.copy()
                scores["diagnostic_cutoff"] = str(cutoff.date())
                scores["diagnostic_seed"] = seed
                score_frames.append(scores)
                if args.save_scores:
                    scores.to_parquet(
                        output / f"scores_{cutoff.date()}_seed{seed}.parquet", index=False
                    )
        if not run_metrics:
            raise RuntimeError("No diagnostic runs were executed")
        aggregate = _aggregate(run_metrics, score_frames)
        payload = {
            "schema_version": "autoencoder-diagnostic-v3",
            "candidate": candidate,
            "cutoffs": [str(value.date()) for value in cutoffs],
            "seeds": seeds,
            "runs": run_metrics,
            "aggregate": aggregate,
            "status": "complete",
            "environment": environment_metadata(requested_device=args.device),
        }
        payload["artifact_manifest"] = {
            "fingerprint": fingerprint,
            "outputs": output_fingerprints(
                    output,
                    (
                        f"scores_{cutoff.date()}_seed{seed}.parquet"
                        for cutoff in cutoffs
                        for seed in seeds
                    )
                    if args.save_scores
                    else (),
                ),
            "result_body": result_body_manifest(payload),
        }
        write_json(result_path, payload)
        print(json.dumps(payload["aggregate"], indent=2))
    except Exception as exc:
        failure = {
            "schema_version": "autoencoder-diagnostic-v3",
            "candidate": candidate,
            "status": "failed",
            "fingerprint": fingerprint,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
