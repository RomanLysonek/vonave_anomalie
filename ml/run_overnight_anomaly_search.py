"""Resumable multi-stage anomaly search for Apple-Silicon overnight execution.

Stages
------
1. diagnostic
   Broad GPU autoencoder search across architectures, representations, windows,
   temporal training spans and calibration policies. Candidates are ranked by
   temporal calibration, seed stability, regime-drift resistance and whether
   their origin score predicts next-week seasonal-baseline difficulty.
2. proxy
   Exact leakage-safe direct panels evaluated with DynamicRidge. Statistical,
   autoencoder and hybrid feature/weight policies compete against a control.
3. neural
   Top proxy candidates are retrained with the actual MPS NeuralNet estimator.
4. confirmation
   The strongest candidates receive wider origins, multiple NN seeds and
   origin-level bootstrap uncertainty. Only this stage can recommend promotion.

Every candidate runs in its own subprocess and writes an atomic result file.
Re-running with ``--resume`` continues from the first missing trial.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from anomaly_search_common import (
    PROFILES,
    SearchProfile,
    autoencoder_action_variants,
    benchmark_origins,
    bootstrap_origin_improvement,
    control_candidate,
    development_diagnostic_boundary,
    development_origins,
    generate_autoencoder_candidates,
    generate_statistical_candidates,
    load_json,
    make_hybrid_candidate,
    selected_forecasting_config,
    validate_target_roles,
    validate_development_diagnostic_boundary,
    write_json,
)
from framework import Config, load_raw
from artifact_provenance import (
    artifact_fingerprint,
    diagnostic_trial_fingerprint,
    environment_metadata,
    forecast_trial_fingerprint,
    load_validated_result,
    OVERNIGHT_SEARCH_SOURCE_PATHS,
)


SCHEMA_VERSION = "overnight-anomaly-search-v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="overnight")
    parser.add_argument(
        "--stage",
        choices=["all", "diagnostic", "proxy", "neural", "confirmation"],
        default="all",
    )
    parser.add_argument("--output-dir", default="outputs/overnight_anomaly_search")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--confirm-recompute-stale",
        action="store_true",
        help="Deliberately replace stale, corrupt, or unverifiable expensive artifacts",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-diagnostic-scores", action="store_true")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=0.0,
        help="Optional graceful stop between trials; zero means no time budget",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preflight(device: str, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    train_path = Path("data/train_data.parquet")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}; run from repository root")
    mps_available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    cuda_available = bool(torch.cuda.is_available())
    if device == "mps" and not mps_available:
        raise RuntimeError("--device mps requested, but torch.backends.mps is unavailable")
    if device == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    disk = shutil.disk_usage(output.resolve().anchor or ".")
    payload = {
        "timestamp": _utc_now(),
        "python": sys.version,
        "torch": torch.__version__,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "requested_device": device,
        "resolved_device": (
            "cuda"
            if device == "auto" and cuda_available
            else "mps"
            if device == "auto" and mps_available
            else "cpu"
            if device == "auto"
            else device
        ),
        "disk_free_gib": disk.free / (1024**3),
        "cpu_count": os.cpu_count(),
        "train_path": str(train_path),
    }
    write_json(output / "preflight.json", payload)
    return payload


def _diagnostic_cutoffs(
    train: pd.DataFrame, count: int, boundary: dict[str, Any]
) -> pd.DatetimeIndex:
    input_end = pd.Timestamp(boundary["diagnostic_input_end"])
    dates = pd.DatetimeIndex(
        sorted(
            pd.to_datetime(
                train.loc[pd.to_datetime(train["DateKey"]) <= input_end, "DateKey"]
            ).unique()
        )
    )
    if dates.empty:
        raise ValueError("No diagnostic cutoff is available before the boundary")
    earliest_position = min(len(dates) - 1, max(0, int(len(dates) * 0.68)))
    eligible = dates[earliest_position:]
    if count >= len(eligible):
        return eligible
    positions = np.linspace(0, len(eligible) - 1, count).round().astype(int)
    return pd.DatetimeIndex(eligible[np.unique(positions)])


def _candidate_file(root: Path, candidate: dict[str, Any]) -> Path:
    path = root / "candidates" / f"{candidate['id']}.json"
    write_json(path, candidate)
    return path


def _run_command(
    command: list[str],
    *,
    log_path: Path,
    dry_run: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] $ {' '.join(command)}\n")
        log.flush()
        if dry_run:
            return 0
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get(
                    "PYTORCH_ENABLE_MPS_FALLBACK", "1"
                ),
            },
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return int(process.wait())


def _should_skip(
    trial_dir: Path,
    args: argparse.Namespace,
    expected_fingerprint: dict[str, Any],
    *,
    required_outputs: tuple[str, ...] = (),
) -> bool:
    result_path = trial_dir / "result.json"
    if result_path.exists():
        payload, reason = load_validated_result(
            result_path,
            expected_fingerprint,
            required_outputs=required_outputs,
        )
        if payload is not None:
            return True
        if not getattr(args, "confirm_recompute_stale", False):
            raise RuntimeError(
                f"Stale or unverifiable result {result_path}: {reason}. "
                "Pass --confirm-recompute-stale to deliberately rerun it."
            )
        print(f"[resume] confirmed recompute of invalid {result_path}: {reason}")
    if (trial_dir / "failure.json").exists():
        failure_path = trial_dir / "failure.json"
        failure = load_json(failure_path)
        if failure.get("fingerprint") == expected_fingerprint and not args.retry_failed:
            return True
        if (
            failure.get("fingerprint") != expected_fingerprint
            and not getattr(args, "confirm_recompute_stale", False)
        ):
            raise RuntimeError(
                f"Stale failure record {failure_path}. Pass "
                "--confirm-recompute-stale to deliberately rerun it."
            )
        print(f"[resume] confirmed retry after failure {failure_path}")
    return False


def _time_budget_exhausted(started: float, max_hours: float) -> bool:
    return max_hours > 0.0 and (time.monotonic() - started) >= max_hours * 3600.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _load_completed_results(
    stage_dir: Path,
    expected_by_id: dict[str, dict[str, Any]],
    *,
    required_outputs: tuple[str, ...] = (),
    confirm_recompute_stale: bool = False,
) -> list[dict[str, Any]]:
    results = []
    if not stage_dir.exists():
        return results
    for result_path in sorted(stage_dir.glob("*/result.json")):
        expected = expected_by_id.get(result_path.parent.name)
        if expected is None:
            print(f"[resume] ignoring unrecognized result {result_path}")
            continue
        payload, reason = load_validated_result(
            result_path, expected, required_outputs=required_outputs
        )
        if payload is None:
            if not confirm_recompute_stale:
                raise RuntimeError(
                    f"Stale or unverifiable result {result_path}: {reason}. "
                    "Pass --confirm-recompute-stale before continuing broad work."
                )
            print(f"[resume] confirmed ignore of invalid {result_path}: {reason}")
            continue
        payload["_result_path"] = str(result_path)
        results.append(payload)
    return results


def _diagnostic_fingerprints(
    candidates: list[dict[str, Any]],
    train: pd.DataFrame,
    cutoffs: pd.DatetimeIndex,
    seeds: tuple[int, ...],
    args: argparse.Namespace,
    boundary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    development_source = validate_development_diagnostic_boundary(
        train, boundary, cutoffs
    )
    return {
        item["id"]: diagnostic_trial_fingerprint(
            candidate=item,
            train_data=development_source,
            cutoffs=cutoffs,
            seeds=seeds,
            device=args.device,
            save_scores=args.save_diagnostic_scores,
            diagnostic_boundary=boundary,
        )
        for item in candidates
    }


def _forecast_fingerprints(
    candidates: list[dict[str, Any]],
    train: pd.DataFrame,
    *,
    model: str,
    epochs: int,
    seeds: tuple[int, ...],
    dev_origins: pd.DatetimeIndex,
    bench_origins: pd.DatetimeIndex,
    device: str,
) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: forecast_trial_fingerprint(
            candidate=item,
            train_data=train,
            model=model,
            epochs=epochs,
            seeds=seeds,
            device=device,
            development_origins=dev_origins,
            benchmark_origins=bench_origins,
        )
        for item in candidates
    }


def _profile_target_roles(
    train: pd.DataFrame,
    profile: SearchProfile,
    diagnostic_boundary: dict[str, Any],
) -> dict[str, Any]:
    """Validate every profile stage before any candidate subprocess can start."""
    cfg = selected_forecasting_config()
    diagnostic_benchmark = benchmark_origins(
        train,
        max(
            profile.proxy_benchmark_origins,
            profile.neural_benchmark_origins,
            profile.confirmation_benchmark_origins,
        ),
        cfg,
    )
    stages = {
        "diagnostic": validate_target_roles(
            calibration_origins=_diagnostic_cutoffs(
                train, profile.autoencoder_cutoffs, diagnostic_boundary
            ),
            benchmark_origins=diagnostic_benchmark,
            horizon=cfg.horizon,
        )
    }
    for name, development_count, benchmark_count in (
        ("proxy", profile.proxy_development_origins, profile.proxy_benchmark_origins),
        ("neural", profile.neural_development_origins, profile.neural_benchmark_origins),
        (
            "confirmation",
            profile.confirmation_development_origins,
            profile.confirmation_benchmark_origins,
        ),
    ):
        stages[name] = validate_target_roles(
            development_origins=development_origins(development_count),
            benchmark_origins=benchmark_origins(train, benchmark_count, cfg),
            horizon=cfg.horizon,
        )
    return {
        "schema_version": "search-target-role-validation-v1",
        "profile": profile.name,
        "stages": stages,
        "content_sha256": artifact_fingerprint(
            schema_version="search-target-role-validation-v1",
            semantic=stages,
        )["semantic_hash"],
    }


def _load_forecast_stage(
    root: Path,
    stage_name: str,
    train: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    model: str,
    epochs: int,
    seeds: tuple[int, ...],
    dev_origins: pd.DatetimeIndex,
    bench_origins: pd.DatetimeIndex,
    device: str,
    confirm_recompute_stale: bool = False,
) -> list[dict[str, Any]]:
    expected = _forecast_fingerprints(
        candidates,
        train,
        model=model,
        epochs=epochs,
        seeds=seeds,
        dev_origins=dev_origins,
        bench_origins=bench_origins,
        device=device,
    )
    return _load_completed_results(
        root / stage_name,
        expected,
        required_outputs=("development_oof.parquet", "benchmark_oof.parquet"),
        confirm_recompute_stale=confirm_recompute_stale,
    )


def _run_diagnostics(
    root: Path,
    train: pd.DataFrame,
    candidates: list[dict[str, Any]],
    profile: SearchProfile,
    args: argparse.Namespace,
    started: float,
    boundary: dict[str, Any],
) -> list[dict[str, Any]]:
    cutoffs = _diagnostic_cutoffs(train, profile.autoencoder_cutoffs, boundary)
    cutoff_arg = ",".join(str(value.date()) for value in cutoffs)
    seed_arg = ",".join(str(value) for value in profile.autoencoder_seeds)
    stage_dir = root / "diagnostic"
    expected_by_id = _diagnostic_fingerprints(
        candidates, train, cutoffs, profile.autoencoder_seeds, args, boundary
    )
    required_outputs = (
        tuple(
            f"scores_{cutoff.date()}_seed{seed}.parquet"
            for cutoff in cutoffs
            for seed in profile.autoencoder_seeds
        )
        if args.save_diagnostic_scores
        else ()
    )
    for index, item in enumerate(candidates, start=1):
        if _time_budget_exhausted(started, args.max_hours):
            break
        trial_dir = stage_dir / item["id"]
        if _should_skip(
            trial_dir,
            args,
            expected_by_id[item["id"]],
            required_outputs=required_outputs,
        ):
            continue
        candidate_path = _candidate_file(root, item)
        command = [
            sys.executable,
            "ml/run_autoencoder_diagnostic_trial.py",
            "--candidate",
            str(candidate_path),
            "--output-dir",
            str(trial_dir),
            "--cutoffs",
            cutoff_arg,
            "--diagnostic-boundary",
            str(root / "diagnostic_boundary.json"),
            "--seeds",
            seed_arg,
            "--device",
            args.device,
        ]
        if args.save_diagnostic_scores:
            command.append("--save-scores")
        print(f"\n[diagnostic {index}/{len(candidates)}] {item['name']} ({item['id']})")
        code = _run_command(
            command, log_path=trial_dir / "trial.log", dry_run=args.dry_run
        )
        if code != 0 and args.fail_fast:
            raise RuntimeError(f"Diagnostic candidate {item['id']} failed with exit {code}")

    results = _load_completed_results(
        stage_dir,
        expected_by_id,
        required_outputs=required_outputs,
        confirm_recompute_stale=args.confirm_recompute_stale,
    )
    rows = []
    for payload in results:
        aggregate = payload["aggregate"]
        means = aggregate["means"]
        rows.append(
            {
                "candidate_id": payload["candidate"]["id"],
                "name": payload["candidate"]["name"],
                "objective": aggregate["diagnostic_objective"],
                "seed_stability": aggregate["seed_score_stability"],
                "future_wape_spearman": means["future_seven_day_wape_spearman"],
                "future_error_top_decile_lift": means[
                    "future_error_top_score_decile_lift"
                ],
                "calibration_far_error": means["calibration_far_error"],
                "holdout_far_error": means["holdout_far_error"],
                "time_drift_spearman": means["time_drift_spearman"],
                "lag1_autocorrelation": means["lag1_score_autocorrelation"],
            }
        )
    rows.sort(key=lambda row: row["objective"], reverse=True)
    _write_csv(root / "diagnostic_leaderboard.csv", rows)
    return results


def _forecast_result_row(payload: dict[str, Any], control: dict[str, Any] | None) -> dict[str, Any]:
    dev = payload["development"]["global"]["WAPE"]
    row = {
        "candidate_id": payload["candidate"]["id"],
        "name": payload["candidate"]["name"],
        "family": payload["candidate"]["family"],
        "model": payload["model"],
        "development_WAPE": dev,
        "development_BiasRatio": payload["development"]["global"]["BiasRatio"],
        "development_top_decile_WAPE": payload["development"]["top_actual_decile"][
            "WAPE"
        ],
    }
    if control is not None:
        control_dev = control["development"]["global"]["WAPE"]
        row["development_relative_improvement"] = (control_dev - dev) / control_dev
    else:
        row["development_relative_improvement"] = 0.0
    row["development_selection_score"] = row["development_relative_improvement"]
    return row


def _rank_forecast_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control = next(
        (payload for payload in results if payload["candidate"]["family"] == "control"),
        None,
    )
    rows = [_forecast_result_row(payload, control) for payload in results]
    rows.sort(
        key=lambda row: (
            -row["development_selection_score"],
            row["development_WAPE"],
            row["candidate_id"],
        )
    )
    return rows


def _add_frozen_benchmark_reporting(
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {payload["candidate"]["id"]: payload for payload in results}
    control = next(
        (payload for payload in results if payload["candidate"]["family"] == "control"),
        None,
    )
    control_wape = (
        float(control["benchmark"]["global"]["WAPE"]) if control is not None else None
    )
    for row in rows:
        benchmark = by_id[row["candidate_id"]]["benchmark"]
        benchmark_wape = float(benchmark["global"]["WAPE"])
        row.update({
            "benchmark_WAPE": benchmark_wape,
            "benchmark_BiasRatio": benchmark["global"]["BiasRatio"],
            "benchmark_top_decile_WAPE": benchmark["top_actual_decile"]["WAPE"],
            "benchmark_relative_change": (
                (benchmark_wape - control_wape) / control_wape
                if control_wape is not None and control_wape > 0
                else 0.0
            ),
        })
    return rows


def _run_forecast_candidates(
    root: Path,
    stage_name: str,
    train: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    model: str,
    dev_origins: pd.DatetimeIndex,
    bench_origins: pd.DatetimeIndex,
    epochs: int,
    seeds: tuple[int, ...],
    args: argparse.Namespace,
    started: float,
) -> list[dict[str, Any]]:
    stage_dir = root / stage_name
    dev_arg = ",".join(str(value.date()) for value in dev_origins)
    bench_arg = ",".join(str(value.date()) for value in bench_origins)
    seeds_arg = ",".join(str(value) for value in seeds)
    expected_by_id = _forecast_fingerprints(
        candidates,
        train,
        model=model,
        epochs=epochs,
        seeds=seeds,
        dev_origins=dev_origins,
        bench_origins=bench_origins,
        device=args.device,
    )
    required_outputs = ("development_oof.parquet", "benchmark_oof.parquet")
    for index, item in enumerate(candidates, start=1):
        if _time_budget_exhausted(started, args.max_hours):
            break
        trial_dir = stage_dir / item["id"]
        if _should_skip(
            trial_dir,
            args,
            expected_by_id[item["id"]],
            required_outputs=required_outputs,
        ):
            continue
        candidate_path = _candidate_file(root, item)
        command = [
            sys.executable,
            "ml/run_anomaly_forecast_trial.py",
            "--candidate",
            str(candidate_path),
            "--output-dir",
            str(trial_dir),
            "--model",
            model,
            "--development-origins",
            dev_arg,
            "--benchmark-origins",
            bench_arg,
            "--epochs",
            str(epochs),
            "--seeds",
            seeds_arg,
            "--device",
            args.device,
            "--cache-dir",
            str(root / "autoencoder_cache"),
        ]
        if args.resume:
            command.append("--resume")
        if args.confirm_recompute_stale:
            command.append("--confirm-recompute-stale")
        print(f"\n[{stage_name} {index}/{len(candidates)}] {item['name']} ({item['id']})")
        code = _run_command(
            command, log_path=trial_dir / "trial.log", dry_run=args.dry_run
        )
        if code != 0 and args.fail_fast:
            raise RuntimeError(f"{stage_name} candidate {item['id']} failed with exit {code}")
    results = _load_completed_results(
        stage_dir,
        expected_by_id,
        required_outputs=required_outputs,
        confirm_recompute_stale=args.confirm_recompute_stale,
    )
    rows = _rank_forecast_results(results)
    _write_csv(
        root / f"{stage_name}_leaderboard.csv",
        _add_frozen_benchmark_reporting(rows, results),
    )
    return results


def _candidate_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in candidates}


def _top_candidates(
    results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    count: int,
    *,
    include_control: bool = True,
) -> list[dict[str, Any]]:
    rows = _rank_forecast_results(results)
    mapping = _candidate_by_id(candidates)
    selected: list[dict[str, Any]] = []
    if include_control:
        control = next((item for item in candidates if item["family"] == "control"), None)
        if control is not None:
            selected.append(control)
    for row in rows:
        if row["family"] == "control":
            continue
        item = mapping.get(row["candidate_id"])
        if item is None:
            continue
        selected.append(item)
        if len(selected) >= count + int(include_control):
            break
    return selected


def _generate_proxy_candidates(
    ae_candidates: list[dict[str, Any]],
    diagnostic_results: list[dict[str, Any]],
    profile: SearchProfile,
    seed: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        diagnostic_results,
        key=lambda payload: payload["aggregate"]["diagnostic_objective"],
        reverse=True,
    )
    top_ids = [payload["candidate"]["id"] for payload in ranked[: profile.autoencoder_proxy_top]]
    mapping = _candidate_by_id(ae_candidates)
    if not top_ids:
        top_ids = [item["id"] for item in ae_candidates[: profile.autoencoder_proxy_top]]
    autoencoder = []
    for candidate_id in top_ids:
        item = mapping.get(candidate_id)
        if item is not None:
            autoencoder.extend(autoencoder_action_variants(item))
    statistical = generate_statistical_candidates(profile.statistical_trials, seed=seed + 17)
    return [control_candidate(), *statistical, *autoencoder]


def _hybrid_candidates_from_proxy(
    proxy_results: list[dict[str, Any]],
    proxy_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _rank_forecast_results(proxy_results)
    mapping = _candidate_by_id(proxy_candidates)
    statistical = [
        mapping[row["candidate_id"]]
        for row in rows
        if row["family"] == "statistical" and row["candidate_id"] in mapping
    ][:3]
    autoencoder = [
        mapping[row["candidate_id"]]
        for row in rows
        if row["family"] == "autoencoder" and row["candidate_id"] in mapping
    ][:3]
    hybrids = []
    index = 1
    for stat in statistical:
        for ae in autoencoder:
            hybrids.append(make_hybrid_candidate(stat, ae, index=index))
            index += 1
    return hybrids


def _load_oof(result: dict[str, Any], split: str) -> pd.DataFrame:
    result_path = Path(result["_result_path"])
    return pd.read_parquet(result_path.parent / f"{split}_oof.parquet")


def _confirmation_recommendation(
    root: Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    control = next(
        payload for payload in results if payload["candidate"]["family"] == "control"
    )
    control_dev = control["development"]["global"]["WAPE"]
    control_top = control["development"]["top_actual_decile"]["WAPE"]
    comparisons = []
    for payload in results:
        if payload is control:
            continue
        dev = payload["development"]["global"]["WAPE"]
        top = payload["development"]["top_actual_decile"]["WAPE"]
        dev_improvement = (control_dev - dev) / control_dev
        top_change = (top - control_top) / control_top
        bootstrap = bootstrap_origin_improvement(
            _load_oof(control, "development"),
            _load_oof(payload, "development"),
            "NeuralNet",
        )
        control_holiday = control["development"].get("by_stratum", {}).get(
            "holiday_event", {}
        ).get("WAPE")
        candidate_holiday = payload["development"].get("by_stratum", {}).get(
            "holiday_event", {}
        ).get("WAPE")
        holiday_change = 0.0
        if control_holiday and candidate_holiday and control_holiday > 0:
            holiday_change = (candidate_holiday - control_holiday) / control_holiday
        passes = {
            "development": dev_improvement >= 0.002,
            "top_decile": top_change <= 0.03,
            "holiday_event": holiday_change <= 0.05,
            "bootstrap_probability": bootstrap[
                "probability_improvement_positive"
            ] >= 0.75,
        }
        comparisons.append(
            {
                "candidate": payload["candidate"],
                "development_WAPE": dev,
                "development_relative_improvement": dev_improvement,
                "development_top_decile_relative_change": top_change,
                "holiday_event_relative_change": holiday_change,
                "bootstrap": bootstrap,
                "passes": passes,
                "accepted": all(passes.values()),
            }
        )
    accepted = [row for row in comparisons if row["accepted"]]
    accepted.sort(
        key=lambda row: (
            -row["development_relative_improvement"],
            row["development_top_decile_relative_change"],
            -row["bootstrap"]["probability_improvement_positive"],
            row["candidate"]["id"],
        )
    )
    winner = accepted[0]["candidate"] if accepted else control["candidate"]
    control_bench = control["benchmark"]["global"]["WAPE"]
    result_by_id = {item["candidate"]["id"]: item for item in results}
    for row in comparisons:
        benchmark_wape = result_by_id[row["candidate"]["id"]]["benchmark"]["global"]["WAPE"]
        row["benchmark_WAPE"] = benchmark_wape
        row["benchmark_relative_change"] = (
            benchmark_wape - control_bench
        ) / control_bench
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "selection_model": "NeuralNet",
        "control": control["candidate"],
        "control_metrics": {
            "development_WAPE": control_dev,
            "benchmark_WAPE": control_bench,
            "development_top_decile_WAPE": control_top,
        },
        "gates": {
            "minimum_development_relative_improvement": 0.002,
            "maximum_top_decile_relative_regression": 0.03,
            "maximum_holiday_event_relative_regression": 0.05,
            "minimum_bootstrap_probability_positive": 0.75,
        },
        "selection_protocol": (
            "development_and_cross_fitted_development_only; "
            "benchmark_is_frozen_reporting_only"
        ),
        "comparisons": comparisons,
        "winner": winner,
        "promote_anomaly_layer": winner["family"] != "control",
        "status": "archived",
        "provenance_status": "unverified",
        "execution_enabled": False,
        "execution_reason": (
            "Archived search evidence is not an independently authenticated "
            "JSON-only recommendation contract."
        ),
    }
    write_json(root / "recommendation.json", payload)
    winner_path = root / "winner_candidate.json"
    if winner_path.exists():
        winner_path.unlink()
    return payload


def _write_report(root: Path, profile: SearchProfile, recommendation: dict[str, Any]) -> None:
    winner = recommendation["winner"]
    lines = [
        "# Overnight anomaly-search result",
        "",
        f"- Profile: `{profile.name}`",
        f"- Generated: `{recommendation['generated_at']}`",
        f"- Winner: `{winner['name']}` (`{winner['id']}`)",
        f"- Promote anomaly layer: `{str(recommendation['promote_anomaly_layer']).lower()}`",
        "",
        "## Promotion gates",
        "",
        "A candidate is promoted only from development evidence: it must improve",
        "development WAPE by at least 0.2%, not regress development top-demand-decile",
        "WAPE by more than 3%, not regress development holiday/event WAPE by more than 5%, and",
        "has at least 75% origin-bootstrap probability of a positive improvement.",
        "The frozen benchmark is displayed after selection and is reporting-only.",
        "",
        "## Candidate comparison",
        "",
        "| Candidate | Dev improvement | Benchmark change | Top-decile change | Bootstrap P(>0) | Accepted |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in recommendation["comparisons"]:
        lines.append(
            "| {name} | {dev:.3%} | {bench:.3%} | {top:.3%} | {prob:.1%} | {accepted} |".format(
                name=row["candidate"]["name"],
                dev=row["development_relative_improvement"],
                bench=row["benchmark_relative_change"],
                top=row["development_top_decile_relative_change"],
                prob=row["bootstrap"]["probability_improvement_positive"],
                accepted="yes" if row["accepted"] else "no",
            )
        )
    lines += [
        "",
        "This result is archived, unverified evidence and is not executable. No winner",
        "configuration or pipeline command is generated. Fold OOF predictions, logs,",
        "checkpoints, caches and leaderboards remain available only for audit.",
    ]
    (root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _invalidate_legacy_execution_artifacts(root: Path) -> None:
    """Fail closed before dry-run, stage-only, validation, or crash exits."""
    for name in ("recommendation.json", "winner_candidate.json"):
        path = root / name
        if path.exists():
            path.unlink()
    write_json(root / "execution_disabled.json", {
        "schema_version": "overnight-execution-disabled-v1",
        "execution_enabled": False,
        "reason": (
            "Overnight evidence has no independently trusted JSON-only execution contract."
        ),
    })


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _invalidate_legacy_execution_artifacts(root)
    profile = PROFILES[args.profile]
    preflight = _preflight(args.device, root)
    train, _ = load_raw(Config())
    frozen_benchmark_count = max(
        profile.proxy_benchmark_origins,
        profile.neural_benchmark_origins,
        profile.confirmation_benchmark_origins,
    )
    diagnostic_boundary = development_diagnostic_boundary(
        train,
        benchmark_origins(
            train, frozen_benchmark_count, selected_forecasting_config()
        ),
    )
    target_role_validation = _profile_target_roles(
        train, profile, diagnostic_boundary
    )
    write_json(root / "diagnostic_boundary.json", diagnostic_boundary)
    ae_candidates = generate_autoencoder_candidates(
        profile.autoencoder_trials,
        seed=args.seed,
        epoch_cap=profile.autoencoder_epoch_cap,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "profile": asdict(profile),
        "arguments": vars(args),
        "preflight": preflight,
        "autoencoder_candidates": ae_candidates,
        "diagnostic_boundary": diagnostic_boundary,
        "target_role_validation": target_role_validation,
        "environment": environment_metadata(
            requested_device=args.device,
            resolved_device=preflight["resolved_device"],
        ),
        "artifact_fingerprint": artifact_fingerprint(
            schema_version=SCHEMA_VERSION,
            semantic={
                "profile": asdict(profile),
                "seed": args.seed,
                "device": args.device,
                "autoencoder_candidates": ae_candidates,
                "diagnostic_boundary": diagnostic_boundary,
                "target_role_validation": target_role_validation,
            },
            dataframes={"train": train},
            source_paths=OVERNIGHT_SEARCH_SOURCE_PATHS,
        ),
    }
    write_json(root / "manifest.json", manifest)

    diagnostic_results: list[dict[str, Any]] = []
    if args.stage in {"all", "diagnostic"}:
        diagnostic_results = _run_diagnostics(
            root, train, ae_candidates, profile, args, started, diagnostic_boundary
        )
        if args.stage == "diagnostic" or _time_budget_exhausted(started, args.max_hours):
            return
    else:
        diagnostic_cutoffs = _diagnostic_cutoffs(
            train, profile.autoencoder_cutoffs, diagnostic_boundary
        )
        diagnostic_results = _load_completed_results(
            root / "diagnostic",
            _diagnostic_fingerprints(
                ae_candidates,
                train,
                diagnostic_cutoffs,
                profile.autoencoder_seeds,
                args,
                diagnostic_boundary,
            ),
            required_outputs=(
                tuple(
                    f"scores_{cutoff.date()}_seed{seed}.parquet"
                    for cutoff in diagnostic_cutoffs
                    for seed in profile.autoencoder_seeds
                )
                if args.save_diagnostic_scores
                else ()
            ),
            confirm_recompute_stale=args.confirm_recompute_stale,
        )

    proxy_candidates = _generate_proxy_candidates(
        ae_candidates, diagnostic_results, profile, args.seed
    )
    write_json(root / "proxy_candidates.json", proxy_candidates)
    proxy_results: list[dict[str, Any]] = []
    if args.stage in {"all", "proxy"}:
        proxy_results = _run_forecast_candidates(
            root,
            "proxy",
            train,
            proxy_candidates,
            model="DynamicRidge",
            dev_origins=development_origins(profile.proxy_development_origins),
            bench_origins=benchmark_origins(
                train, profile.proxy_benchmark_origins, selected_forecasting_config()
            ),
            epochs=1,
            seeds=(42,),
            args=args,
            started=started,
        )
        hybrids = _hybrid_candidates_from_proxy(proxy_results, proxy_candidates)
        if hybrids:
            proxy_candidates.extend(hybrids)
            write_json(root / "proxy_candidates.json", proxy_candidates)
            proxy_results = _run_forecast_candidates(
                root,
                "proxy",
                train,
                proxy_candidates,
                model="DynamicRidge",
                dev_origins=development_origins(profile.proxy_development_origins),
                bench_origins=benchmark_origins(
                    train, profile.proxy_benchmark_origins, selected_forecasting_config()
                ),
                epochs=1,
                seeds=(42,),
                args=args,
                started=started,
            )
        if args.stage == "proxy" or _time_budget_exhausted(started, args.max_hours):
            return
    else:
        saved_proxy_candidates = root / "proxy_candidates.json"
        if saved_proxy_candidates.exists():
            proxy_candidates = load_json(saved_proxy_candidates)
        proxy_dev = development_origins(profile.proxy_development_origins)
        proxy_bench = benchmark_origins(
            train, profile.proxy_benchmark_origins, selected_forecasting_config()
        )
        proxy_results = _load_forecast_stage(
            root,
            "proxy",
            train,
            proxy_candidates,
            model="DynamicRidge",
            epochs=1,
            seeds=(42,),
            dev_origins=proxy_dev,
            bench_origins=proxy_bench,
            device=args.device,
            confirm_recompute_stale=args.confirm_recompute_stale,
        )

    neural_candidates = _top_candidates(
        proxy_results, proxy_candidates, profile.neural_top, include_control=True
    )
    write_json(root / "neural_candidates.json", neural_candidates)
    neural_results: list[dict[str, Any]] = []
    if args.stage in {"all", "neural"}:
        neural_results = _run_forecast_candidates(
            root,
            "neural",
            train,
            neural_candidates,
            model="NeuralNet",
            dev_origins=development_origins(profile.neural_development_origins),
            bench_origins=benchmark_origins(
                train, profile.neural_benchmark_origins, selected_forecasting_config()
            ),
            epochs=profile.neural_epochs,
            seeds=profile.neural_seeds,
            args=args,
            started=started,
        )
        if args.stage == "neural" or _time_budget_exhausted(started, args.max_hours):
            return
    else:
        saved_neural = root / "neural_candidates.json"
        if saved_neural.exists():
            neural_candidates = load_json(saved_neural)
        neural_results = _load_forecast_stage(
            root,
            "neural",
            train,
            neural_candidates,
            model="NeuralNet",
            epochs=profile.neural_epochs,
            seeds=profile.neural_seeds,
            dev_origins=development_origins(profile.neural_development_origins),
            bench_origins=benchmark_origins(
                train, profile.neural_benchmark_origins, selected_forecasting_config()
            ),
            device=args.device,
            confirm_recompute_stale=args.confirm_recompute_stale,
        )

    confirmation_candidates = _top_candidates(
        neural_results,
        neural_candidates,
        profile.confirmation_top,
        include_control=True,
    )
    write_json(root / "confirmation_candidates.json", confirmation_candidates)
    confirmation_results = _run_forecast_candidates(
        root,
        "confirmation",
        train,
        confirmation_candidates,
        model="NeuralNet",
        dev_origins=development_origins(profile.confirmation_development_origins),
        bench_origins=benchmark_origins(
            train, profile.confirmation_benchmark_origins, selected_forecasting_config()
        ),
        epochs=profile.confirmation_epochs,
        seeds=profile.confirmation_seeds,
        args=args,
        started=started,
    )
    if args.dry_run or not confirmation_results:
        return
    recommendation = _confirmation_recommendation(root, confirmation_results)
    _write_report(root, profile, recommendation)
    print(json.dumps({
        "winner": recommendation["winner"],
        "promote_anomaly_layer": recommendation["promote_anomaly_layer"],
        "report": str(root / "FINAL_REPORT.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
