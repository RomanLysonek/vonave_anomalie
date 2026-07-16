"""Train the weekend-v2 experts on all history and create the final blend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_search_common import apply_candidate_config, load_json, selected_forecasting_config, write_json
from framework import Config, compute_baseline, load_raw
from models.naive_baselines import moving_average_predict, seasonal_naive_predict
from pipeline import run_final_forecast_direct
from weekend_v2_common import apply_meta_model, apply_specialist_gate, apply_weight_plan, load_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendation", required=True)
    parser.add_argument("--output-dir", default="outputs/weekend_v2_search/final")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", default="42,123,777")
    parser.add_argument("--resume-members", action="store_true")
    return parser.parse_args()


def active_members(recommendation: dict) -> set[str]:
    winner = recommendation["winner"]
    plan = winner["plan"]
    method = plan["method"]
    if method == "control":
        return {plan["member"]}
    if method in {"ridge_residual", "risk_gate"}:
        return {item["column"] for item in recommendation["members"]}
    if method == "specialist_gate":
        return {plan["control_column"], plan["specialist_column"]}
    if method in {"global_convex", "aggregate_reconciled"}:
        return {key for key, value in plan["weights"].items() if float(value) > 1e-8}
    if method == "horizon_convex":
        active = {
            key for key, value in plan["global_weights"].items() if float(value) > 1e-8
        }
        for weights in plan["horizon_weights"].values():
            active.update(key for key, value in weights.items() if float(value) > 1e-8)
        return active
    if method == "product_convex":
        active = {
            key for key, value in plan["global_weights"].items() if float(value) > 1e-8
        }
        for weights in plan["product_weights"].values():
            active.update(key for key, value in weights.items() if float(value) > 1e-8)
        return active
    raise ValueError(f"Unknown final plan method: {method}")


def main() -> None:
    args = parse_args()
    recommendation_path = Path(args.recommendation)
    recommendation = load_json(recommendation_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, test = load_raw(Config())
    seeds = tuple(int(token) for token in args.seeds.split(",") if token)
    needed = active_members(recommendation)

    frame = test[["ProductId", "DateKey"]].copy().reset_index(drop=True)
    last_train_date = pd.Timestamp(train["DateKey"].max())
    frame["horizon"] = (pd.to_datetime(frame["DateKey"]) - last_train_date).dt.days.astype(int)
    frame["origin"] = last_train_date
    frame["ProductAvailable"] = True
    frame["baseline"] = compute_baseline(test, train, "weighted_4321")
    frame["pred_SeasonalNaive"] = seasonal_naive_predict(test, train, lag_days=7)
    frame["pred_MovingAvg28"] = moving_average_predict(test, train, window=28)

    trained = []
    for item in recommendation["members"]:
        column = item["column"]
        if column not in needed:
            continue
        candidate = item["candidate"]
        member_dir = output / "members" / candidate["id"]
        cached = member_dir / "predictions.csv"
        if args.resume_members and cached.exists():
            cached_frame = pd.read_csv(cached)
            predictions = cached_frame["prediction"].to_numpy(dtype=float)
            diagnostics = {
                column: cached_frame[column].to_numpy(dtype=float)
                for column in cached_frame.columns
                if column.startswith("anomaly_") or column.startswith("autoencoder_")
            }
        else:
            cfg = selected_forecasting_config()
            apply_candidate_config(cfg, candidate)
            cfg.final_epochs = args.epochs
            cfg.seeds = seeds
            cfg.autoencoder_device = args.device
            cfg.autoencoder_cache_dir = str(output / "autoencoder_cache")
            cfg.output_dir = str(member_dir)
            _, predictions, diagnostics = run_final_forecast_direct(
                train, test, cfg, return_diagnostics=True
            )
            member_dir.mkdir(parents=True, exist_ok=True)
            cached_frame = pd.DataFrame({
                "ProductId": frame["ProductId"],
                "DateKey": frame["DateKey"],
                "prediction": predictions,
            })
            for key, values in diagnostics.items():
                if key.startswith("anomaly_") or key.startswith("autoencoder_"):
                    cached_frame[key] = np.asarray(values, dtype=float)
            cached_frame.to_csv(cached, index=False)
            write_json(member_dir / "candidate.json", candidate)
        frame[column] = np.asarray(predictions, dtype=float)
        for key, values in diagnostics.items():
            if key == "prediction":
                continue
            if key.startswith("anomaly_") or key.startswith("autoencoder_"):
                feature_key = f"feature__{candidate['id']}__{key}"
                frame[feature_key] = np.asarray(values, dtype=float)
        trained.append(candidate["id"])

    winner = recommendation["winner"]
    plan = winner["plan"]
    method = plan["method"]
    member_columns = [item["column"] for item in recommendation["members"] if item["column"] in frame]
    if method == "control":
        prediction = frame[plan["member"]].to_numpy(dtype=float)
    elif method in {
        "global_convex", "horizon_convex", "product_convex", "aggregate_reconciled"
    }:
        prediction = apply_weight_plan(frame, plan, member_columns)
    elif method in {"ridge_residual", "risk_gate", "specialist_gate"}:
        model_path = Path(plan["model_path"])
        if not model_path.is_absolute() and not model_path.exists():
            model_path = recommendation_path.parent / "ensemble" / model_path.name
        bundle = load_pickle(model_path)
        if method == "specialist_gate":
            prediction = apply_specialist_gate(frame, bundle)
        else:
            prediction = apply_meta_model(frame, bundle)
    else:
        raise ValueError(method)

    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, None)
    frame["prediction_weekend_v2_raw"] = prediction
    frame["prediction_weekend_v2"] = np.rint(prediction).astype(int)
    submission = frame[["ProductId", "DateKey"]].copy()
    submission["Quantity"] = frame["prediction_weekend_v2"]
    submission.to_csv(output / "submission.csv", index=False)
    submission.to_parquet(output / "submission.parquet", index=False)
    frame.to_csv(output / "final_member_forecasts.csv", index=False)
    frame.to_parquet(output / "final_member_forecasts.parquet", index=False)
    write_json(output / "run_metadata.json", {
        "recommendation": str(recommendation_path),
        "winner": winner,
        "trained_member_ids": trained,
        "epochs": args.epochs,
        "seeds": list(seeds),
        "rows": len(frame),
    })
    print(json.dumps({
        "winner": winner["name"],
        "method": method,
        "submission": str(output / "submission.csv"),
        "trained_members": trained,
    }, indent=2))


if __name__ == "__main__":
    main()
