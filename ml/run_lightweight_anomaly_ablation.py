"""Low-memory causal anomaly-weight ablation on the seasonal baseline.

This is a first-stage falsification test for restricted environments: it uses
the exact availability-aware 4:3:2:1 same-weekday baseline and changes only
the weights of lag observations flagged by the DAVID-inspired detector. It
does not replace full direct-panel model confirmation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_detection import build_demand_anomaly_profile
from framework import Config, compute_baseline, compute_metrics, reindex_daily_calendar
from offline_parquet import read_parquet

BASE_LAGS = (7, 14, 21, 28)
BASE_WEIGHTS = np.array([4.0, 3.0, 2.0, 1.0], dtype=float)
DEV_ORIGINS = pd.to_datetime([
    "2023-01-10",
    "2024-06-20",
    "2024-11-29",
    "2025-02-10",
])


def load_data() -> pd.DataFrame:
    train = read_parquet("data/train_data.parquet")
    train["Quantity"] = (train["QuantityApp"] + train["QuantityWeb"]).astype(float)
    return reindex_daily_calendar(train)


def weighted_baseline(target: pd.DataFrame, history: pd.DataFrame,
                      profile: pd.DataFrame) -> np.ndarray:
    hist = history[["ProductId", "DateKey", "Quantity", "ProductAvailable"]].merge(
        profile[["ProductId", "DateKey", "anomaly_weight"]],
        on=["ProductId", "DateKey"], how="left", validate="one_to_one",
    )
    available = hist["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
    hist["usable_quantity"] = pd.to_numeric(hist["Quantity"], errors="coerce").where(available)
    hist["anomaly_weight"] = pd.to_numeric(hist["anomaly_weight"], errors="coerce").fillna(1.0)
    q_lookup = hist.set_index(["ProductId", "DateKey"])["usable_quantity"]
    w_lookup = hist.set_index(["ProductId", "DateKey"])["anomaly_weight"]

    predictions = np.full(len(target), np.nan, dtype=float)
    for row_index, row in enumerate(target.itertuples(index=False)):
        quantities = []
        weights = []
        for lag, base_weight in zip(BASE_LAGS, BASE_WEIGHTS):
            key = (row.ProductId, row.DateKey - pd.Timedelta(days=lag))
            quantity = q_lookup.get(key, np.nan)
            if pd.notna(quantity):
                quantities.append(float(quantity))
                weights.append(float(base_weight) * float(w_lookup.get(key, 1.0)))
        if weights and sum(weights) > 0:
            predictions[row_index] = float(np.dot(quantities, weights) / np.sum(weights))
    return predictions


def score(frame: pd.DataFrame, prediction: str) -> dict:
    mask = frame["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
    mask &= frame["actual"].notna() & frame[prediction].notna()
    return compute_metrics(
        frame.loc[mask, "actual"].to_numpy(float),
        frame.loc[mask, prediction].to_numpy(float),
    )


def main() -> None:
    output = Path("outputs/lightweight_anomaly_ablation")
    output.mkdir(parents=True, exist_ok=True)
    train = load_data()
    max_date = train["DateKey"].max()
    benchmark_origins = pd.DatetimeIndex([
        max_date - pd.Timedelta(days=7 * i) for i in range(1, 5)
    ])
    policies = {
        "weight_soft": {"strength": 0.50, "min_weight": 0.40},
        "weight_default": {"strength": 1.00, "min_weight": 0.20},
    }

    rows = []
    fold_metrics = []
    for split, origins in (("development", DEV_ORIGINS), ("benchmark", benchmark_origins)):
        for origin in origins:
            history = train[train["DateKey"] <= origin].copy()
            evaluation = train[train["DateKey"].between(
                origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=7)
            )].copy()
            evaluation = evaluation[["ProductId", "DateKey", "Quantity", "ProductAvailable"]]
            evaluation = evaluation.rename(columns={"Quantity": "actual"})
            evaluation["origin"] = origin
            evaluation["split"] = split
            evaluation["pred_control"] = compute_baseline(
                evaluation.rename(columns={"actual": "Quantity"}), history,
                baseline_variant="weighted_4321",
            )

            for name, params in policies.items():
                cfg = Config()
                cfg.anomaly_mode = "weight"
                cfg.anomaly_weight_strength = params["strength"]
                cfg.anomaly_min_weight = params["min_weight"]
                profile, _ = build_demand_anomaly_profile(history, cfg)
                evaluation[f"pred_{name}"] = weighted_baseline(evaluation, history, profile)

            for model in ["control", *policies]:
                metric = score(evaluation, f"pred_{model}")
                fold_metrics.append({
                    "split": split,
                    "origin": str(origin.date()),
                    "policy": model,
                    **{k: metric[k] for k in ("WAPE", "MAE", "BiasRatio", "n")},
                })
            rows.append(evaluation)
            print(split, origin.date(), {m: fold_metrics[-(len(policies)+1)+i]["WAPE"] for i,m in enumerate(["control",*policies])})

    predictions = pd.concat(rows, ignore_index=True)
    metrics = pd.DataFrame(fold_metrics)
    summary_rows = []
    for split in ("development", "benchmark"):
        subset = predictions[predictions["split"] == split]
        for model in ["control", *policies]:
            metric = score(subset, f"pred_{model}")
            summary_rows.append({
                "split": split, "policy": model,
                **{k: metric[k] for k in ("WAPE", "MAE", "BiasRatio", "n")},
            })
    summary = pd.DataFrame(summary_rows)
    control = summary[summary["policy"] == "control"].set_index("split")
    comparisons = []
    for model in policies:
        candidate = summary[summary["policy"] == model].set_index("split")
        dev_improvement = (
            control.loc["development", "WAPE"] - candidate.loc["development", "WAPE"]
        ) / control.loc["development", "WAPE"]
        benchmark_change = (
            candidate.loc["benchmark", "WAPE"] - control.loc["benchmark", "WAPE"]
        ) / control.loc["benchmark", "WAPE"]
        comparisons.append({
            "policy": model,
            "development_relative_improvement": float(dev_improvement),
            "benchmark_relative_change": float(benchmark_change),
            "passes_development_gate": bool(dev_improvement >= 0.002),
            "passes_benchmark_guard": bool(benchmark_change <= 0.02),
        })
    accepted = [x for x in comparisons if x["passes_development_gate"] and x["passes_benchmark_guard"]]
    winner = max(accepted, key=lambda x: x["development_relative_improvement"])["policy"] if accepted else "control"
    payload = {
        "test": "availability-aware weighted_4321 baseline anomaly-lag ablation",
        "development_origins": [str(x.date()) for x in DEV_ORIGINS],
        "benchmark_origins": [str(x.date()) for x in benchmark_origins],
        "summary": summary.to_dict(orient="records"),
        "comparisons": comparisons,
        "winner": winner,
        "interpretation": "First-stage falsification only; full direct NeuralNet confirmation is required before promotion.",
    }
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "result.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
