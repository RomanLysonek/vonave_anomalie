"""Generate retrospective demand anomalies and forecast-time context risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from anomaly_detection import build_demand_anomaly_profile
from context_risk import ContextRiskDetector
from framework import Config, load_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/anomaly_audit")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--mode", choices=["weight", "features", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.anomaly_mode = args.mode
    cfg.anomaly_evt_alpha = args.alpha
    train, test = load_raw(cfg)

    profile, metadata = build_demand_anomaly_profile(train, cfg)
    profile.to_csv(output / "demand_anomaly_profile.csv", index=False)

    detector = ContextRiskDetector(random_state=cfg.seed).fit(train)
    train_context = detector.score(train)
    test_context = detector.score(test)
    test_context.to_csv(output / "test_context_risk.csv", index=False)

    daily = (
        test_context.groupby("DateKey", as_index=False)
        .agg(
            mean_context_risk=("context_risk_percentile", "mean"),
            max_context_risk=("context_risk_percentile", "max"),
            shifted_products=("context_shift_flag", "sum"),
        )
    )
    daily.to_csv(output / "test_context_risk_daily.csv", index=False)

    metadata["test_context"] = {
        "n_rows": int(len(test_context)),
        "n_shift_flags": int(test_context["context_shift_flag"].sum()),
        "max_percentile": float(test_context["context_risk_percentile"].max()),
        "mean_percentile": float(test_context["context_risk_percentile"].mean()),
    }
    with open(output / "anomaly_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"Wrote anomaly audit to {output}")


if __name__ == "__main__":
    main()
