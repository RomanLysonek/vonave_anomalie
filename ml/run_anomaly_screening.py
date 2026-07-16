"""Direct-first DAVID anomaly ablation on the frozen forecasting pipeline.

The screening stage uses LightGBM because it exercises the exact same direct
panel and sample-weight contract at a fraction of the neural-network cost.  A
candidate is only recommended for full NN confirmation if it improves the
seasonally scattered development origins without materially regressing the
recent pseudo-test benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from framework import Config, compute_metrics, load_raw
from pipeline import DEVELOPMENT_ORIGINS, recent_benchmark_origins, run_walk_forward_cv_direct


SCREEN_DEVELOPMENT_ORIGINS = pd.to_datetime([
    "2023-01-10",
    "2024-06-20",
    "2024-11-29",
    "2025-02-10",
])


@dataclass(frozen=True)
class Candidate:
    name: str
    mode: str
    evt_alpha: float = 0.01
    weight_strength: float = 1.0
    min_weight: float = 0.20


CANDIDATES = (
    Candidate("control", "off"),
    Candidate("weight_soft", "weight", weight_strength=0.50, min_weight=0.40),
    Candidate("weight_default", "weight", weight_strength=1.00, min_weight=0.20),
    Candidate("features_only", "features"),
    Candidate("both_soft", "both", weight_strength=0.50, min_weight=0.40),
    Candidate("both_default", "both", weight_strength=1.00, min_weight=0.20),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "screen", "full"], default="screen")
    parser.add_argument("--output-dir", default="outputs/anomaly_screening")
    parser.add_argument("--candidates", default="all", help="Comma-separated names or 'all'")
    parser.add_argument("--benchmark-tolerance", type=float, default=0.02)
    parser.add_argument("--min-relative-improvement", type=float, default=0.002)
    return parser.parse_args()


def _final_selected_config() -> Config:
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
    cfg.seeds = (42,)
    cfg.cv_epochs = 12
    cfg.final_epochs = 12
    return cfg


def _origins(profile: str, train: pd.DataFrame, cfg: Config) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    benchmark = recent_benchmark_origins(train, cfg)
    if profile == "smoke":
        return SCREEN_DEVELOPMENT_ORIGINS[-2:], benchmark[:2]
    if profile == "screen":
        return SCREEN_DEVELOPMENT_ORIGINS, benchmark
    return DEVELOPMENT_ORIGINS, benchmark


def _score(oof: pd.DataFrame, prediction_column: str = "pred_LightGBM") -> dict:
    if oof.empty:
        return {"WAPE": float("nan"), "MAE": float("nan"), "BiasRatio": float("nan"), "n": 0}
    available = oof.get("ProductAvailable", pd.Series(True, index=oof.index))
    mask = available.astype("boolean").fillna(False).astype(bool)
    mask &= pd.to_numeric(oof["actual"], errors="coerce").notna()
    mask &= pd.to_numeric(oof[prediction_column], errors="coerce").notna()
    metrics = compute_metrics(
        oof.loc[mask, "actual"].to_numpy(dtype=float),
        oof.loc[mask, prediction_column].to_numpy(dtype=float),
    )
    return {key: metrics[key] for key in ("WAPE", "MAE", "BiasRatio", "n")}


def _stratum_rows(oof: pd.DataFrame, candidate: str, split: str) -> list[dict]:
    rows = []
    global_metrics = _score(oof)
    rows.append({"candidate": candidate, "split": split, "stratum": "global", **global_metrics})
    if "validation_stratum" in oof:
        for stratum, frame in oof.groupby("validation_stratum", sort=True):
            rows.append({
                "candidate": candidate,
                "split": split,
                "stratum": str(stratum),
                **_score(frame),
            })
    return rows


def _select(candidates: list[dict], tolerance: float, min_improvement: float) -> dict:
    global_rows = {
        (row["candidate"], row["split"]): row
        for row in candidates if row["stratum"] == "global"
    }
    control_dev = global_rows[("control", "development")]["WAPE"]
    control_bench = global_rows[("control", "benchmark")]["WAPE"]
    eligible = []
    for name in sorted({row["candidate"] for row in candidates} - {"control"}):
        dev = global_rows[(name, "development")]["WAPE"]
        bench = global_rows[(name, "benchmark")]["WAPE"]
        dev_improvement = (control_dev - dev) / control_dev
        benchmark_change = (bench - control_bench) / control_bench
        eligible.append({
            "candidate": name,
            "development_WAPE": dev,
            "benchmark_WAPE": bench,
            "development_relative_improvement": dev_improvement,
            "benchmark_relative_change": benchmark_change,
            "passes_development_gate": dev_improvement >= min_improvement,
            "passes_benchmark_guard": benchmark_change <= tolerance,
        })
    accepted = [
        row for row in eligible
        if row["passes_development_gate"] and row["passes_benchmark_guard"]
    ]
    accepted.sort(key=lambda row: (row["development_WAPE"], row["benchmark_WAPE"]))
    winner = accepted[0]["candidate"] if accepted else "control"
    return {
        "control_development_WAPE": control_dev,
        "control_benchmark_WAPE": control_bench,
        "candidate_comparison": eligible,
        "winner": winner,
        "accepted": winner != "control",
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    base_cfg = _final_selected_config()
    train, _ = load_raw(base_cfg)
    development_origins, benchmark_origins = _origins(args.profile, train, base_cfg)

    if args.candidates == "all":
        selected = list(CANDIDATES)
    else:
        requested = {token.strip() for token in args.candidates.split(",") if token.strip()}
        selected = [candidate for candidate in CANDIDATES if candidate.name in requested]
        missing = requested - {candidate.name for candidate in selected}
        if missing:
            raise ValueError(f"Unknown candidates: {sorted(missing)}")
        if not any(candidate.name == "control" for candidate in selected):
            selected.insert(0, CANDIDATES[0])

    metric_rows: list[dict] = []
    all_oof = []
    for candidate in selected:
        cfg = copy.deepcopy(base_cfg)
        cfg.anomaly_mode = candidate.mode
        cfg.anomaly_evt_alpha = candidate.evt_alpha
        cfg.anomaly_weight_strength = candidate.weight_strength
        cfg.anomaly_min_weight = candidate.min_weight
        print(f"\n=== {candidate.name}: {asdict(candidate)} ===")
        for split, origins in (
            ("development", development_origins),
            ("benchmark", benchmark_origins),
        ):
            oof = run_walk_forward_cv_direct(
                train,
                origins,
                split,
                cfg,
                run_neural=False,
                structured_models=("LightGBM",),
            )
            oof["candidate"] = candidate.name
            oof["screen_split"] = split
            all_oof.append(oof)
            metric_rows.extend(_stratum_rows(oof, candidate.name, split))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "anomaly_screening_metrics.csv", index=False)
    if all_oof:
        pd.concat(all_oof, ignore_index=True).to_csv(
            output / "anomaly_screening_oof.csv", index=False
        )

    selection = _select(
        metric_rows,
        tolerance=args.benchmark_tolerance,
        min_improvement=args.min_relative_improvement,
    )
    candidate_map = {candidate.name: asdict(candidate) for candidate in selected}
    winner = candidate_map[selection["winner"]]
    command = (
        "caffeinate -i uv run python ml/pipeline.py "
        "--forecast-strategy direct --primary-strategy direct "
        "--submission-model NeuralNet --selection-metric WAPE "
        "--selection-protocol test-aligned --training-window-days all "
        "--recency-half-life-days none --baseline-variant weighted_4321 "
        "--trend-features off "
        "--c2-feature-groups price,campaign,lifecycle,market,event "
        "--c34-config outputs/c34_screening/recommendation.json "
        f"--anomaly-mode {winner['mode']} "
        f"--anomaly-evt-alpha {winner['evt_alpha']} "
        f"--anomaly-weight-strength {winner['weight_strength']} "
        f"--anomaly-min-weight {winner['min_weight']} "
        "--nn-batch-size 512 --nn-lr-scaling fixed --reset-checkpoints"
    )
    payload = {
        "schema_version": "david-anomaly-screen-v1",
        "profile": args.profile,
        "development_origins": [str(date.date()) for date in development_origins],
        "benchmark_origins": [str(date.date()) for date in benchmark_origins],
        "selection_policy": {
            "model": "LightGBM",
            "metric": "conditional global WAPE",
            "min_relative_improvement": args.min_relative_improvement,
            "benchmark_tolerance": args.benchmark_tolerance,
        },
        "candidates": candidate_map,
        "selection": selection,
        "full_nn_confirmation_command": command,
    }
    with open(output / "recommendation.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
