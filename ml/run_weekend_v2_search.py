"""Evidence-driven weekend-v2 search.

Unlike the first overnight funnel, every promotion stage uses the NeuralNet.
Anomaly modes remain available as specialist generators, while the final stage
searches leakage-safe blends, horizon mixtures, aggregate reconciliation and
nonlinear residual gates across confirmed experts.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from anomaly_search_common import (
    benchmark_origins,
    development_origins,
    load_json,
    selected_forecasting_config,
    write_json,
)
from framework import Config, load_raw
from weekend_v2_common import (
    WEEKEND_V2_PROFILES,
    WeekendV2Profile,
    apply_meta_model,
    apply_specialist_gate,
    apply_weight_plan,
    bootstrap_probability,
    crossfit_meta_model,
    crossfit_plan,
    crossfit_specialist_gate,
    evaluate_prediction,
    fit_aggregate_reconciliation,
    candidate,
    control_candidate,
    generate_regime_specialists,
    generate_statistical_neighborhood,
    load_top_autoencoder_specialists,
    merge_oof,
    save_pickle,
    search_convex_weights,
    wape,
)
from artifact_provenance import (
    artifact_fingerprint,
    config_hash,
    environment_metadata,
    file_fingerprint,
    file_hash,
    forecast_trial_fingerprint,
    load_validated_result,
    WEEKEND_SEARCH_SOURCE_PATHS,
)

SCHEMA_VERSION = "weekend-v2-search-v4"
RECOMMENDATION_MANIFEST_SCHEMA_VERSION = "weekend-v2-recommendation-provenance-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(WEEKEND_V2_PROFILES), default="weekend-v2")
    parser.add_argument(
        "--stage", choices=["all", "screen", "refine", "confirmation", "ensemble"],
        default="all",
    )
    parser.add_argument("--prior-root", default="outputs/overnight_anomaly_search")
    parser.add_argument("--output-dir", default="outputs/weekend_v2_search")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--confirm-recompute-stale",
        action="store_true",
        help="Deliberately replace stale, corrupt, or unverifiable expensive artifacts",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-hours", type=float, default=0.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_path(root: Path, item: dict[str, Any]) -> Path:
    path = root / "candidates" / f"{item['id']}.json"
    write_json(path, item)
    return path


def run_command(command: list[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] $ {' '.join(command)}\n")
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
                "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1"),
            },
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return int(process.wait())


def exhausted(started: float, max_hours: float) -> bool:
    return max_hours > 0 and time.monotonic() - started >= max_hours * 3600


def rank_development_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates using development evidence; benchmark is reporting-only."""
    control = next(x for x in results if x["candidate"]["family"] == "control")
    control_dev = float(control["development"]["global"]["WAPE"])
    control_top = float(control["development"]["top_actual_decile"]["WAPE"])
    rows: list[dict[str, Any]] = []
    for payload in results:
        dev = float(payload["development"]["global"]["WAPE"])
        dev_improvement = (control_dev - dev) / control_dev
        top = float(payload["development"]["top_actual_decile"]["WAPE"])
        top_improvement = (control_top - top) / control_top
        control_origins = control["development"].get("by_origin", {})
        candidate_origins = payload["development"].get("by_origin", {})
        origin_improvements = [
            (float(control_origins[key]["WAPE"]) - float(candidate_origins[key]["WAPE"]))
            / float(control_origins[key]["WAPE"])
            for key in sorted(set(control_origins) & set(candidate_origins))
            if float(control_origins[key]["WAPE"]) > 0
        ]
        stability = float(np.std(origin_improvements)) if origin_improvements else 1.0
        positive_share = (
            float(np.mean(np.asarray(origin_improvements) > 0))
            if origin_improvements else 0.0
        )
        selection_score = (
            dev_improvement
            + 0.10 * top_improvement
            + 0.05 * positive_share
            - 0.12 * stability
        )
        rows.append({
            "candidate_id": payload["candidate"]["id"],
            "name": payload["candidate"]["name"],
            "family": payload["candidate"]["family"],
            "development_WAPE": dev,
            "development_relative_improvement": dev_improvement,
            "top_decile_relative_improvement": top_improvement,
            "origin_improvement_std": stability,
            "origin_positive_share": positive_share,
            "development_selection_score": selection_score,
        })
    return sorted(
        rows,
        key=lambda row: (-row["development_selection_score"], row["candidate_id"]),
    )


def add_frozen_benchmark_reporting(
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {payload["candidate"]["id"]: payload for payload in results}
    control = next(x for x in results if x["candidate"]["family"] == "control")
    control_wape = float(control["benchmark"]["global"]["WAPE"])
    for row in rows:
        benchmark_wape = float(
            by_id[row["candidate_id"]]["benchmark"]["global"]["WAPE"]
        )
        row["benchmark_WAPE"] = benchmark_wape
        row["benchmark_relative_improvement"] = (
            control_wape - benchmark_wape
        ) / control_wape
    return rows


def generate_development_only_candidates(
    profile: WeekendV2Profile, *, seed: int, prior_root: Path
) -> list[dict[str, Any]]:
    """Build the pool without using prior benchmark metrics."""
    no_prior = prior_root / ".no_legacy_selection_input"
    statistical = generate_statistical_neighborhood(
        profile.statistical_candidates, seed=seed, prior_root=no_prior
    )
    regimes = generate_regime_specialists(profile.regime_candidates)
    autoencoders = load_top_autoencoder_specialists(prior_root, profile.autoencoder_top)
    hybrids: list[dict[str, Any]] = []
    for stat in statistical[:2]:
        for autoencoder in autoencoders[:2]:
            cfg = {**stat["config"], **autoencoder["config"]}
            cfg.update({"anomaly_mode": "both", "anomaly_source": "hybrid"})
            hybrids.append(candidate(
                "hybrid",
                f"v2_hybrid_{stat['id']}_{autoencoder['id']}",
                cfg,
                parents=[stat["id"], autoencoder["id"]],
                diagnostic=autoencoder.get("diagnostic"),
            ))
    by_id = {
        item["id"]: item
        for item in [control_candidate(), *statistical, *regimes, *autoencoders, *hybrids]
    }
    return list(by_id.values())


def select_development_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    accepted = [row for row in rows if row["accepted"]]
    if not accepted:
        return None
    return sorted(
        accepted,
        key=lambda row: (-row["selection_score"], row["name"]),
    )[0]


def _stage_fingerprints(
    train: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    dev_origins: pd.DatetimeIndex,
    bench_origins: pd.DatetimeIndex,
    epochs: int,
    seeds: tuple[int, ...],
    device: str,
) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: forecast_trial_fingerprint(
            candidate=item,
            train_data=train,
            model="NeuralNet",
            epochs=epochs,
            seeds=seeds,
            device=device,
            development_origins=dev_origins,
            benchmark_origins=bench_origins,
        )
        for item in candidates
    }


def _load_valid_stage_results(
    stage_dir: Path,
    expected_by_id: dict[str, dict[str, Any]],
    *,
    confirm_recompute_stale: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not stage_dir.exists():
        return results
    for result_path in sorted(stage_dir.glob("*/result.json")):
        expected = expected_by_id.get(result_path.parent.name)
        if expected is None:
            print(f"[resume] ignoring unrecognized result {result_path}")
            continue
        payload, reason = load_validated_result(
            result_path,
            expected,
            required_outputs=("development_oof.parquet", "benchmark_oof.parquet"),
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


def run_stage(
    root: Path,
    name: str,
    train: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    dev_origins: pd.DatetimeIndex,
    bench_origins: pd.DatetimeIndex,
    epochs: int,
    seeds: tuple[int, ...],
    args: argparse.Namespace,
    started: float,
) -> list[dict[str, Any]]:
    stage_dir = root / name
    dev_arg = ",".join(str(value.date()) for value in dev_origins)
    bench_arg = ",".join(str(value.date()) for value in bench_origins)
    seed_arg = ",".join(str(value) for value in seeds)
    expected_by_id = _stage_fingerprints(
        train,
        candidates,
        dev_origins=dev_origins,
        bench_origins=bench_origins,
        epochs=epochs,
        seeds=seeds,
        device=args.device,
    )
    for index, item in enumerate(candidates, 1):
        if exhausted(started, args.max_hours):
            break
        trial = stage_dir / item["id"]
        result_path = trial / "result.json"
        invalid_result = False
        if result_path.exists():
            payload, reason = load_validated_result(
                result_path,
                expected_by_id[item["id"]],
                required_outputs=("development_oof.parquet", "benchmark_oof.parquet"),
            )
            if payload is not None:
                continue
            if not args.confirm_recompute_stale:
                raise RuntimeError(
                    f"Stale or unverifiable result {result_path}: {reason}. "
                    "Pass --confirm-recompute-stale to deliberately rerun it."
                )
            print(f"[resume] confirmed recompute of invalid {result_path}: {reason}")
            invalid_result = True
        failure_path = trial / "failure.json"
        if failure_path.exists():
            failure = load_json(failure_path)
            if (
                failure.get("fingerprint") == expected_by_id[item["id"]]
                and not args.retry_failed
            ):
                continue
            if (
                failure.get("fingerprint") != expected_by_id[item["id"]]
                and not args.confirm_recompute_stale
            ):
                raise RuntimeError(
                    f"Stale failure record {failure_path}. Pass "
                    "--confirm-recompute-stale to deliberately rerun it."
                )
            print(f"[resume] confirmed retry after failure {failure_path}")
        command = [
            sys.executable,
            "ml/run_anomaly_forecast_trial.py",
            "--candidate", str(candidate_path(root, item)),
            "--output-dir", str(trial),
            "--model", "NeuralNet",
            "--development-origins", dev_arg,
            "--benchmark-origins", bench_arg,
            "--epochs", str(epochs),
            "--seeds", seed_arg,
            "--device", args.device,
            "--cache-dir", str(root / "autoencoder_cache"),
        ]
        if args.resume:
            command.append("--resume")
        if args.confirm_recompute_stale:
            command.append("--confirm-recompute-stale")
        print(f"\n[{name} {index}/{len(candidates)}] {item['name']} ({item['id']})")
        code = run_command(command, trial / "trial.log", args.dry_run)
        if code != 0 and args.fail_fast:
            raise RuntimeError(f"{name} trial {item['id']} failed with exit {code}")
    results = _load_valid_stage_results(
        stage_dir,
        expected_by_id,
        confirm_recompute_stale=args.confirm_recompute_stale,
    )
    if results:
        ranked = rank_development_results(results)
        write_csv(
            root / f"{name}_leaderboard.csv",
            add_frozen_benchmark_reporting(ranked, results),
        )
    return results


def select_candidates(
    results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top: int,
) -> list[dict[str, Any]]:
    mapping = {item["id"]: item for item in candidates}
    rows = rank_development_results(results)
    selected: list[dict[str, Any]] = []
    control = next(item for item in candidates if item["family"] == "control")
    selected.append(control)
    # First take the best candidates, then preserve one credible member from
    # each family so complementarity is not destroyed by a single scalar rank.
    for row in rows:
        if row["family"] == "control":
            continue
        item = mapping.get(row["candidate_id"])
        if item and item not in selected:
            selected.append(item)
        if len(selected) >= top + 1:
            break
    represented = {item["family"] for item in selected}
    for family in ("statistical", "regime", "autoencoder", "hybrid"):
        if family in represented:
            continue
        row = next((r for r in rows if r["family"] == family), None)
        if row and row["candidate_id"] in mapping:
            selected.append(mapping[row["candidate_id"]])
    return selected


def select_diverse_candidates(
    results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top: int,
    *,
    samples: int,
    seed: int,
    report_path: Path,
) -> list[dict[str, Any]]:
    """Greedily promote experts by cross-fitted marginal ensemble value.

    Standalone WAPE is deliberately not the only criterion: a weak specialist
    can be valuable when its residuals differ from those of the control.
    """
    mapping = {item["id"]: item for item in candidates}
    development, all_members = merge_oof(results, "development")
    control_result = next(x for x in results if x["candidate"]["family"] == "control")
    control_member = f"member__{control_result['candidate']['id']}"
    selected = [control_member]
    decisions: list[dict[str, Any]] = []
    available = [member for member in all_members if member != control_member]

    while available and len(selected) < top + 1:
        best: tuple[float, str, dict[str, Any]] | None = None
        for option_index, option in enumerate(available):
            subset = [*selected, option]
            crossfit, plan = crossfit_plan(
                development,
                subset,
                method="global_convex",
                samples=max(1200, samples // 20),
                seed=seed + len(selected) * 100 + option_index,
                reference_column=control_member,
            )
            dev_metrics = evaluate_prediction(
                development, crossfit, control_member
            )
            selected_mean = development[selected].mean(axis=1).to_numpy(dtype=float)
            candidate_prediction = development[option].to_numpy(dtype=float)
            actual = development["actual"].to_numpy(dtype=float)
            left = actual - selected_mean
            right = actual - candidate_prediction
            correlation = float(np.corrcoef(left, right)[0, 1])
            if not np.isfinite(correlation):
                correlation = 1.0
            diversity = 1.0 - abs(correlation)
            score = (
                dev_metrics["relative_improvement"]
                - 0.12 * max(dev_metrics["top_decile_relative_change"], 0.0)
                + 0.04 * diversity
            )
            detail = {
                "candidate_id": option.removeprefix("member__"),
                "candidate_name": mapping[option.removeprefix("member__")]["name"],
                "family": mapping[option.removeprefix("member__")]["family"],
                "selected_before": [x.removeprefix("member__") for x in selected],
                "development": dev_metrics,
                "residual_correlation": correlation,
                "diversity": diversity,
                "marginal_score": score,
                "plan": plan,
            }
            if best is None or score > best[0]:
                best = (score, option, detail)
        assert best is not None
        _, chosen, detail = best
        selected.append(chosen)
        available.remove(chosen)
        decisions.append(detail)

    # Guarantee that the expensive confirmation directly tests at least one
    # anomaly-derived expert, unless none completed the refine stage.  This is
    # exploration coverage, not automatic promotion into the final blend.
    anomaly_families = {"statistical", "autoencoder", "hybrid"}
    selected_ids = [member.removeprefix("member__") for member in selected]
    if not any(mapping[item]["family"] in anomaly_families for item in selected_ids):
        ranked = rank_development_results(results)
        anomaly_row = next(
            (row for row in ranked if row["family"] in anomaly_families), None
        )
        if anomaly_row is not None:
            replacement = anomaly_row["candidate_id"]
            if len(selected) >= 2:
                selected[-1] = f"member__{replacement}"
            else:
                selected.append(f"member__{replacement}")
            decisions.append({
                "candidate_id": replacement,
                "candidate_name": mapping[replacement]["name"],
                "family": mapping[replacement]["family"],
                "reason": "anomaly-family confirmation coverage",
            })

    output = [mapping[member.removeprefix("member__")] for member in selected]
    write_json(report_path, {
        "method": "greedy_crossfit_marginal_ensemble_value",
        "selected": [item["id"] for item in output],
        "decisions": decisions,
    })
    return output


def crossfit_aggregate(
    frame: pd.DataFrame,
    members: list[str],
    *,
    samples: int,
    seed: int,
    reference: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.full(len(frame), np.nan, dtype=float)
    origins = sorted(frame["origin"].unique())
    for index, origin in enumerate(origins):
        train = frame[frame["origin"] != origin]
        valid_index = frame.index[frame["origin"] == origin]
        base = search_convex_weights(
            train, members, samples=max(1000, samples // max(len(origins), 1)),
            seed=seed + index, reference_column=reference,
        )
        plan = fit_aggregate_reconciliation(
            train, {"method": "global_convex", "weights": base["weights"]}, members
        )
        output[valid_index] = apply_weight_plan(frame.loc[valid_index], plan, members)
    full_base = search_convex_weights(
        frame, members, samples=samples, seed=seed, reference_column=reference
    )
    full_plan = fit_aggregate_reconciliation(
        frame, {"method": "global_convex", "weights": full_base["weights"]}, members
    )
    return output, full_plan


def _write_recommendation_with_provenance(
    root: Path,
    recommendation: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    result_by_id = {
        result["candidate"]["id"]: result
        for result in results
    }
    member_bindings = []
    for member in recommendation["members"]:
        candidate_id = member["candidate"]["id"]
        result = result_by_id[candidate_id]
        result_manifest = result["artifact_manifest"]
        raw_result_path = result.get("_result_path")
        if not isinstance(raw_result_path, str):
            raise RuntimeError(f"Result path is unavailable for member {candidate_id}")
        result_path = Path(raw_result_path).resolve()
        try:
            relative_result_path = result_path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Member result must stay below the recommendation root: {raw_result_path}"
            ) from exc
        outputs = result_manifest.get("outputs")
        required_oof = {
            name: outputs.get(name) if isinstance(outputs, dict) else None
            for name in ("development_oof.parquet", "benchmark_oof.parquet")
        }
        if any(value is None for value in required_oof.values()):
            raise RuntimeError(f"Member {candidate_id} lacks authenticated OOF outputs")
        member.update({
            "source_result_path": relative_result_path.as_posix(),
            "expected_result_fingerprint": result_manifest["fingerprint"],
            "canonical_result_body_digest": result_manifest["result_body"],
            "candidate_body_sha256": config_hash(member["candidate"]),
            "development_summary_sha256": config_hash(result["development"]),
            "benchmark_summary_sha256": config_hash(result["benchmark"]),
            "oof_output_fingerprints": required_oof,
        })
        member_bindings.append({
            "column": member["column"],
            "candidate_id": candidate_id,
            "candidate_body_sha256": config_hash(member["candidate"]),
            "source_result_path": member["source_result_path"],
            "expected_result_fingerprint": member["expected_result_fingerprint"],
            "canonical_result_body_digest": member["canonical_result_body_digest"],
            "development_summary_sha256": member["development_summary_sha256"],
            "benchmark_summary_sha256": member["benchmark_summary_sha256"],
            "oof_output_fingerprints": member["oof_output_fingerprints"],
        })

    required_pickles: dict[str, dict[str, Any]] = {}
    plan = recommendation["winner"]["plan"]
    if plan["method"] in {"ridge_residual", "risk_gate", "specialist_gate"}:
        relative = Path(plan["model_path"])
        fingerprint = file_fingerprint(root / relative)
        if fingerprint is None:
            raise FileNotFoundError(root / relative)
        required_pickles[relative.as_posix()] = fingerprint

    recommendation["provenance_manifest"] = "recommendation.provenance.json"
    recommendation_path = root / "recommendation.json"
    write_json(recommendation_path, recommendation)
    search_manifest_path = root / "manifest.json"
    search_manifest = load_json(search_manifest_path)
    provenance = {
        "schema_version": RECOMMENDATION_MANIFEST_SCHEMA_VERSION,
        "recommendation_file": file_fingerprint(recommendation_path),
        "recommendation_body_sha256": config_hash(recommendation),
        "search_manifest": {
            "path": "manifest.json",
            "file": file_fingerprint(search_manifest_path),
            "input_fingerprint": search_manifest["artifact_fingerprint"],
        },
        "member_bindings": member_bindings,
        "member_bindings_sha256": config_hash(member_bindings),
        "winner_body_sha256": config_hash(recommendation["winner"]),
        "winner_plan_sha256": config_hash(plan),
        "required_pickles": required_pickles,
    }
    write_json(root / recommendation["provenance_manifest"], provenance)


def ensemble_stage(
    root: Path,
    results: list[dict[str, Any]],
    profile: WeekendV2Profile,
    seed: int,
) -> dict[str, Any]:
    development, members = merge_oof(results, "development")
    id_to_candidate = {result["candidate"]["id"]: result["candidate"] for result in results}
    control_result = next(result for result in results if result["candidate"]["family"] == "control")
    control_column = f"member__{control_result['candidate']['id']}"

    plans: list[dict[str, Any]] = []
    global_cf, global_plan = crossfit_plan(
        development, members, method="global_convex", samples=profile.ensemble_samples,
        seed=seed, reference_column=control_column,
    )
    plans.append({"name": "global_convex", "plan": global_plan, "crossfit": global_cf})

    horizon_cf, horizon_plan = crossfit_plan(
        development, members, method="horizon_convex", samples=profile.ensemble_samples,
        seed=seed + 1000, reference_column=control_column,
    )
    plans.append({"name": "horizon_convex", "plan": horizon_plan, "crossfit": horizon_cf})

    product_cf, product_plan = crossfit_plan(
        development, members, method="product_convex",
        samples=profile.ensemble_samples, seed=seed + 1500,
        reference_column=control_column,
    )
    plans.append({"name": "product_convex", "plan": product_plan, "crossfit": product_cf})

    aggregate_cf, aggregate_plan = crossfit_aggregate(
        development, members, samples=profile.ensemble_samples, seed=seed + 2000,
        reference=control_column,
    )
    plans.append({"name": "aggregate_reconciled", "plan": aggregate_plan, "crossfit": aggregate_cf})

    for offset, kind in enumerate(("ridge_residual", "risk_gate"), start=1):
        crossfit, bundle = crossfit_meta_model(
            development, members, kind=kind, seed=seed + 3000 + offset,
            reference_column=control_column,
        )
        relative_model_path = Path("ensemble") / f"{kind}.pkl"
        save_pickle(root / relative_model_path, bundle)
        plans.append({
            "name": kind,
            "plan": {"method": kind, "model_path": relative_model_path.as_posix()},
            "crossfit": crossfit,
            "bundle": bundle,
            "bundle_kind": kind,
        })

    # Constrained gates test the specific hypothesis suggested by the first
    # overnight run: anomaly experts help only in selected regimes.
    for specialist_index, specialist in enumerate(
        (member for member in members if member != control_column), start=1
    ):
        for max_alpha in (0.50, 0.85):
            crossfit, bundle = crossfit_specialist_gate(
                development, control_column=control_column,
                specialist_column=specialist,
                seed=seed + 4000 + specialist_index,
                max_alpha=max_alpha,
            )
            label = f"specialist_gate_{specialist_index:02d}_a{int(max_alpha * 100):02d}"
            relative_model_path = Path("ensemble") / f"{label}.pkl"
            save_pickle(root / relative_model_path, bundle)
            plans.append({
                "name": label,
                "plan": {
                    "method": "specialist_gate",
                    "model_path": relative_model_path.as_posix(),
                    "control_column": control_column,
                    "specialist_column": specialist,
                },
                "crossfit": crossfit,
                "bundle": bundle,
                "bundle_kind": "specialist_gate",
            })

    rows = []
    for item in plans:
        crossfit = item["crossfit"]
        dev_metrics = evaluate_prediction(development, crossfit, control_column)
        bootstrap = bootstrap_probability(
            development, crossfit, control_column, samples=5000, seed=seed
        )
        holiday = dev_metrics.get("by_stratum", {}).get("holiday_event", {})
        if holiday and holiday.get("reference_WAPE", 0) > 0:
            holiday_change = (
                holiday["WAPE"] - holiday["reference_WAPE"]
            ) / holiday["reference_WAPE"]
        else:
            holiday_change = 0.0
        passes = {
            "development": dev_metrics["relative_improvement"] >= 0.002,
            "top_decile": dev_metrics["top_decile_relative_change"] <= 0.02,
            "holiday_event": holiday_change <= 0.03,
            "bootstrap_probability": bootstrap["probability_improvement_positive"] >= 0.75,
        }
        row = {
            "name": item["name"],
            "plan": item["plan"],
            "development": dev_metrics,
            "bootstrap": bootstrap,
            "holiday_event_relative_change": holiday_change,
            "passes": passes,
            "accepted": all(passes.values()),
            "selection_score": (
                dev_metrics["relative_improvement"]
                - 0.10 * max(dev_metrics["top_decile_relative_change"], 0.0)
                - 0.10 * max(holiday_change, 0.0)
            ),
        }
        rows.append(row)
        # Save row-level predictions for forensic comparison.
        development.assign(pred_weekend_v2=crossfit).to_parquet(
            root / "ensemble" / f"{item['name']}_development_oof.parquet", index=False
        )

    selected_winner = select_development_winner(rows)
    if selected_winner is not None:
        winner = selected_winner
        promote = True
    else:
        winner = {
            "name": "control",
            "plan": {"method": "control", "member": control_column},
            "development": evaluate_prediction(
                development, development[control_column].to_numpy(), control_column
            ),
            "accepted": True,
            "selection_score": 0.0,
        }
        promote = False

    benchmark, benchmark_members = merge_oof(results, "benchmark")
    if members != benchmark_members:
        raise RuntimeError("Development and benchmark member schemas differ")
    for item, row in zip(plans, rows, strict=True):
        if item.get("bundle_kind") == "specialist_gate":
            benchmark_prediction = apply_specialist_gate(benchmark, item["bundle"])
        elif "bundle" in item:
            benchmark_prediction = apply_meta_model(benchmark, item["bundle"])
        else:
            benchmark_prediction = apply_weight_plan(benchmark, item["plan"], members)
        row["benchmark"] = evaluate_prediction(
            benchmark, benchmark_prediction, control_column
        )
        benchmark.assign(pred_weekend_v2=benchmark_prediction).to_parquet(
            root / "ensemble" / f"{item['name']}_benchmark_oof.parquet", index=False
        )
    if not promote:
        winner["benchmark"] = evaluate_prediction(
            benchmark, benchmark[control_column].to_numpy(), control_column
        )

    member_payloads = []
    for member in members:
        candidate_id = member.removeprefix("member__")
        member_payloads.append({
            "column": member,
            "candidate": id_to_candidate[candidate_id],
        })
    recommendation = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "reference_member": control_column,
        "members": member_payloads,
        "comparisons": rows,
        "winner": winner,
        "promote_weekend_v2": promote,
        "selection_protocol": (
            "cross_fitted_development_only; benchmark_is_frozen_reporting_only"
        ),
        "final_submission_command": (
            "caffeinate -dimsu uv run python ml/run_weekend_v2_final.py "
            f"--recommendation {root / 'recommendation.json'} "
            f"--output-dir {root / 'final'} --device mps --resume-members"
        ),
    }
    _write_recommendation_with_provenance(root, recommendation, results)
    write_json(root / "winner_plan.json", winner)
    write_csv(root / "ensemble_leaderboard.csv", [
        {
            "name": row["name"],
            "development_WAPE": row["development"]["WAPE"],
            "development_relative_improvement": row["development"]["relative_improvement"],
            "benchmark_WAPE": row["benchmark"]["WAPE"],
            "benchmark_relative_improvement": row["benchmark"]["relative_improvement"],
            "top_decile_relative_change": row["development"]["top_decile_relative_change"],
            "holiday_event_relative_change": row["holiday_event_relative_change"],
            "bootstrap_probability": row["bootstrap"]["probability_improvement_positive"],
            "accepted": row["accepted"],
            "selection_score": row["selection_score"],
        }
        for row in rows
    ])
    report = [
        "# Weekend-v2 result", "",
        f"- Generated: `{recommendation['generated_at']}`",
        f"- Winner: `{winner['name']}`",
        f"- Promote: `{str(promote).lower()}`", "",
        "## Interpretation", "",
        "Anomaly configurations are treated as complementary experts. A failure to",
        "win as a standalone model does not remove them when their errors diversify",
        "the canonical NeuralNet and the cross-fitted ensemble passes every gate.", "",
        "## Meta-policy comparison", "",
        "| Policy | Dev improvement | Benchmark improvement | Bootstrap P(>0) | Accepted |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['name']} | {row['development']['relative_improvement']:.3%} | "
            f"{row['benchmark']['relative_improvement']:.3%} | "
            f"{row['bootstrap']['probability_improvement_positive']:.1%} | "
            f"{'yes' if row['accepted'] else 'no'} |"
        )
    (root / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return recommendation


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prior_root = Path(args.prior_root)
    profile = WEEKEND_V2_PROFILES[args.profile]
    candidates = generate_development_only_candidates(
        profile, seed=args.seed, prior_root=prior_root
    )
    write_json(root / "candidate_pool.json", candidates)
    write_json(root / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "profile": profile.__dict__,
        "arguments": vars(args),
        "prior_root": str(prior_root),
        "legacy_prior_evidence": {
            "scientific_status": "contaminated",
            "provenance_status": "unverified",
            "selection_use": "excluded",
        },
        "candidate_count": len(candidates),
        "train_file_hash": file_hash("data/train_data.parquet"),
        "environment": environment_metadata(requested_device=args.device),
        "artifact_fingerprint": artifact_fingerprint(
            schema_version=SCHEMA_VERSION,
            semantic={
                "profile": profile.__dict__,
                "seed": args.seed,
                "device": args.device,
                "candidates": candidates,
            },
            source_paths=WEEKEND_SEARCH_SOURCE_PATHS,
        ),
    })
    if args.dry_run:
        write_json(root / "dry_run_plan.json", {
            "candidate_count": len(candidates),
            "families": {
                family: sum(item["family"] == family for item in candidates)
                for family in sorted({item["family"] for item in candidates})
            },
            "profile": profile.__dict__,
            "note": "No training was executed.",
        })
        print(json.dumps(load_json(root / "dry_run_plan.json"), indent=2))
        return

    train, _ = load_raw(Config())
    screen_results: list[dict[str, Any]] = []
    if args.stage in {"all", "screen"}:
        screen_results = run_stage(
            root, "screen", train, candidates,
            dev_origins=development_origins(profile.screen_development_origins),
            bench_origins=benchmark_origins(train, profile.screen_benchmark_origins, selected_forecasting_config()),
            epochs=profile.screen_epochs, seeds=profile.screen_seeds,
            args=args, started=started,
        )
        if args.stage == "screen" or exhausted(started, args.max_hours):
            return
    else:
        screen_dev = development_origins(profile.screen_development_origins)
        screen_bench = benchmark_origins(
            train, profile.screen_benchmark_origins, selected_forecasting_config()
        )
        screen_results = _load_valid_stage_results(
            root / "screen",
            _stage_fingerprints(
                train,
                candidates,
                dev_origins=screen_dev,
                bench_origins=screen_bench,
                epochs=profile.screen_epochs,
                seeds=profile.screen_seeds,
                device=args.device,
            ),
            confirm_recompute_stale=args.confirm_recompute_stale,
        )

    if not screen_results:
        raise RuntimeError(
            "No completed screen results are available. Run --stage screen first or use --stage all."
        )
    refine_candidates = select_diverse_candidates(
        screen_results,
        candidates,
        profile.refine_top,
        samples=max(12000, profile.ensemble_samples // 2),
        seed=args.seed + 4500,
        report_path=root / "refine_selection.json",
    )
    write_json(root / "refine_candidates.json", refine_candidates)
    if args.stage in {"all", "refine"}:
        refine_results = run_stage(
            root, "refine", train, refine_candidates,
            dev_origins=development_origins(profile.refine_development_origins),
            bench_origins=benchmark_origins(train, profile.refine_benchmark_origins, selected_forecasting_config()),
            epochs=profile.refine_epochs, seeds=profile.refine_seeds,
            args=args, started=started,
        )
        if args.stage == "refine" or exhausted(started, args.max_hours):
            return
    else:
        refine_dev = development_origins(profile.refine_development_origins)
        refine_bench = benchmark_origins(
            train, profile.refine_benchmark_origins, selected_forecasting_config()
        )
        refine_results = _load_valid_stage_results(
            root / "refine",
            _stage_fingerprints(
                train,
                refine_candidates,
                dev_origins=refine_dev,
                bench_origins=refine_bench,
                epochs=profile.refine_epochs,
                seeds=profile.refine_seeds,
                device=args.device,
            ),
            confirm_recompute_stale=args.confirm_recompute_stale,
        )

    if not refine_results:
        raise RuntimeError(
            "No completed refine results are available. Run --stage refine first or use --stage all."
        )
    confirmation_candidates = select_diverse_candidates(
        refine_results,
        refine_candidates,
        profile.confirmation_top,
        samples=profile.ensemble_samples,
        seed=args.seed + 5000,
        report_path=root / "confirmation_selection.json",
    )
    write_json(root / "confirmation_candidates.json", confirmation_candidates)
    if args.stage in {"all", "confirmation"}:
        confirmation_results = run_stage(
            root, "confirmation", train, confirmation_candidates,
            dev_origins=development_origins(profile.confirmation_development_origins),
            bench_origins=benchmark_origins(train, profile.confirmation_benchmark_origins, selected_forecasting_config()),
            epochs=profile.confirmation_epochs, seeds=profile.confirmation_seeds,
            args=args, started=started,
        )
        if args.stage == "confirmation" or exhausted(started, args.max_hours):
            return
    else:
        confirmation_dev = development_origins(
            profile.confirmation_development_origins
        )
        confirmation_bench = benchmark_origins(
            train,
            profile.confirmation_benchmark_origins,
            selected_forecasting_config(),
        )
        confirmation_results = _load_valid_stage_results(
            root / "confirmation",
            _stage_fingerprints(
                train,
                confirmation_candidates,
                dev_origins=confirmation_dev,
                bench_origins=confirmation_bench,
                epochs=profile.confirmation_epochs,
                seeds=profile.confirmation_seeds,
                device=args.device,
            ),
            confirm_recompute_stale=args.confirm_recompute_stale,
        )

    if not confirmation_results:
        raise RuntimeError(
            "No completed confirmation results are available. Run --stage confirmation first or use --stage all."
        )
    recommendation = ensemble_stage(root, confirmation_results, profile, args.seed)
    print(json.dumps({
        "winner": recommendation["winner"]["name"],
        "promote_weekend_v2": recommendation["promote_weekend_v2"],
        "report": str(root / "FINAL_REPORT.md"),
        "final_command": recommendation["final_submission_command"],
    }, indent=2))


if __name__ == "__main__":
    main()
