"""Run one anomaly candidate through exact direct-panel forecast evaluation.

The worker supports DynamicRidge proxy screening and MPS NeuralNet promotion.
Each invocation owns one candidate and exits afterwards, which bounds memory
use and makes the overnight orchestrator safely resumable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import pandas as pd

from anomaly_search_common import (
    apply_candidate_config,
    load_json,
    selected_forecasting_config,
    summarize_oof,
    write_json,
)
from framework import load_raw
from pipeline import run_walk_forward_cv_direct
from artifact_provenance import (
    environment_metadata,
    forecast_trial_fingerprint,
    output_fingerprints,
    result_body_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=["DynamicRidge", "NeuralNet"], required=True)
    parser.add_argument("--development-origins", required=True)
    parser.add_argument("--benchmark-origins", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--cache-dir", default="outputs/overnight_anomaly_search/autoencoder_cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-recompute-stale", action="store_true")
    return parser.parse_args()


def _parse_origins(value: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(token) for token in value.split(",") if token])


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate = load_json(Path(args.candidate))
    result_path = output / "result.json"
    fingerprint = None
    try:
        cfg = selected_forecasting_config()
        apply_candidate_config(cfg, candidate)
        cfg.cv_epochs = args.epochs
        cfg.final_epochs = args.epochs
        cfg.seeds = tuple(int(token) for token in args.seeds.split(",") if token)
        cfg.autoencoder_device = args.device
        cfg.autoencoder_cache_dir = args.cache_dir
        cfg.confirm_recompute_stale = args.confirm_recompute_stale
        train, _ = load_raw(cfg)
        development_origins = _parse_origins(args.development_origins)
        benchmark_origins = _parse_origins(args.benchmark_origins)
        fingerprint = forecast_trial_fingerprint(
            candidate=candidate,
            train_data=train,
            model=args.model,
            epochs=args.epochs,
            seeds=cfg.seeds,
            device=args.device,
            development_origins=development_origins,
            benchmark_origins=benchmark_origins,
        )
        checkpoint_root = output / "checkpoints"
        if (
            not args.resume
            and any(checkpoint_root.rglob("*.pkl"))
            and not args.confirm_recompute_stale
        ):
            raise RuntimeError(
                f"Existing fold checkpoints under {checkpoint_root} would be recomputed. "
                "Use --resume to validate/reuse them or pass "
                "--confirm-recompute-stale to deliberately replace them."
            )

        if args.model == "DynamicRidge":
            run_neural = False
            structured_models: tuple[str, ...] = ("DynamicRidge",)
        else:
            run_neural = True
            structured_models = ()

        frames = {}
        timings: list[dict] = []
        for split, origins in (
            ("development", development_origins),
            ("benchmark", benchmark_origins),
        ):
            oof = run_walk_forward_cv_direct(
                train,
                origins,
                split,
                cfg,
                timings=timings,
                checkpoint_dir=str(checkpoint_root),
                resume=args.resume,
                confirm_recompute_stale=args.confirm_recompute_stale,
                run_neural=run_neural,
                structured_models=structured_models,
            )
            frames[split] = oof
            if not oof.empty:
                oof.to_parquet(output / f"{split}_oof.parquet", index=False)

        output_names = [
            f"{split}_oof.parquet"
            for split, frame in frames.items()
            if not frame.empty
        ]
        payload = {
            "schema_version": "anomaly-forecast-trial-v3",
            "candidate": candidate,
            "model": args.model,
            "epochs": args.epochs,
            "seeds": list(cfg.seeds),
            "development_origins": [str(value.date()) for value in development_origins],
            "benchmark_origins": [str(value.date()) for value in benchmark_origins],
            "development": summarize_oof(frames["development"], args.model),
            "benchmark": summarize_oof(frames["benchmark"], args.model),
            "timings": timings,
            "status": "complete",
            "environment": environment_metadata(requested_device=args.device),
        }
        payload["artifact_manifest"] = {
            "fingerprint": fingerprint,
            "outputs": output_fingerprints(output, output_names),
            "result_body": result_body_manifest(payload),
        }
        write_json(result_path, payload)
        print(json.dumps({
            "candidate": candidate["id"],
            "model": args.model,
            "development_WAPE": payload["development"]["global"]["WAPE"],
            "benchmark_WAPE": payload["benchmark"]["global"]["WAPE"],
        }, indent=2))
    except Exception as exc:
        failure = {
            "schema_version": "anomaly-forecast-trial-v3",
            "candidate": candidate,
            "model": args.model,
            "status": "failed",
            "fingerprint": fingerprint,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
