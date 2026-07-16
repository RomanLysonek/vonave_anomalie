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
    generate_weekend_v2_candidates,
    load_stage_results,
    merge_oof,
    rank_forecast_results,
    save_pickle,
    search_convex_weights,
    wape,
)

SCHEMA_VERSION = "weekend-v2-search-v1"


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


def run_stage(
    root: Path,
    name: str,
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
    for index, item in enumerate(candidates, 1):
        if exhausted(started, args.max_hours):
            break
        trial = stage_dir / item["id"]
        if (trial / "result.json").exists():
            continue
        if (trial / "failure.json").exists() and not args.retry_failed:
            continue
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
        print(f"\n[{name} {index}/{len(candidates)}] {item['name']} ({item['id']})")
        code = run_command(command, trial / "trial.log", args.dry_run)
        if code != 0 and args.fail_fast:
            raise RuntimeError(f"{name} trial {item['id']} failed with exit {code}")
    results = load_stage_results(stage_dir)
    if results:
        write_csv(root / f"{name}_leaderboard.csv", rank_forecast_results(results))
    return results


def select_candidates(
    results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top: int,
) -> list[dict[str, Any]]:
    mapping = {item["id"]: item for item in candidates}
    rows = rank_forecast_results(results)
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
    benchmark, benchmark_members = merge_oof(results, "benchmark")
    if all_members != benchmark_members:
        raise RuntimeError("Refine development/benchmark schemas differ")
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
            benchmark_prediction = apply_weight_plan(benchmark, plan, subset)
            bench_metrics = evaluate_prediction(
                benchmark, benchmark_prediction, control_member
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
                + 0.20 * min(bench_metrics["relative_improvement"], 0.05)
                - 1.5 * max(-0.02 - bench_metrics["relative_improvement"], 0.0)
                - 0.12 * max(dev_metrics["top_decile_relative_change"], 0.0)
                + 0.04 * diversity
            )
            detail = {
                "candidate_id": option.removeprefix("member__"),
                "candidate_name": mapping[option.removeprefix("member__")]["name"],
                "family": mapping[option.removeprefix("member__")]["family"],
                "selected_before": [x.removeprefix("member__") for x in selected],
                "development": dev_metrics,
                "benchmark": bench_metrics,
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
        ranked = rank_forecast_results(results)
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


def ensemble_stage(
    root: Path,
    results: list[dict[str, Any]],
    profile: WeekendV2Profile,
    seed: int,
) -> dict[str, Any]:
    development, members = merge_oof(results, "development")
    benchmark, benchmark_members = merge_oof(results, "benchmark")
    if members != benchmark_members:
        raise RuntimeError("Development and benchmark member schemas differ")
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
        model_path = root / "ensemble" / f"{kind}.pkl"
        save_pickle(model_path, bundle)
        plans.append({
            "name": kind,
            "plan": {"method": kind, "model_path": str(model_path)},
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
            model_path = root / "ensemble" / f"{label}.pkl"
            save_pickle(model_path, bundle)
            plans.append({
                "name": label,
                "plan": {
                    "method": "specialist_gate",
                    "model_path": str(model_path),
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
        if item.get("bundle_kind") == "specialist_gate":
            benchmark_prediction = apply_specialist_gate(benchmark, item["bundle"])
        elif "bundle" in item:
            benchmark_prediction = apply_meta_model(benchmark, item["bundle"])
        else:
            benchmark_prediction = apply_weight_plan(benchmark, item["plan"], members)
        dev_metrics = evaluate_prediction(development, crossfit, control_column)
        bench_metrics = evaluate_prediction(benchmark, benchmark_prediction, control_column)
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
            "benchmark": bench_metrics["relative_improvement"] >= -0.01,
            "top_decile": dev_metrics["top_decile_relative_change"] <= 0.02,
            "holiday_event": holiday_change <= 0.03,
            "bootstrap_probability": bootstrap["probability_improvement_positive"] >= 0.75,
        }
        row = {
            "name": item["name"],
            "plan": item["plan"],
            "development": dev_metrics,
            "benchmark": bench_metrics,
            "bootstrap": bootstrap,
            "holiday_event_relative_change": holiday_change,
            "passes": passes,
            "accepted": all(passes.values()),
            "selection_score": (
                dev_metrics["relative_improvement"]
                + 0.35 * bench_metrics["relative_improvement"]
                - 0.10 * max(dev_metrics["top_decile_relative_change"], 0.0)
                - 0.10 * max(holiday_change, 0.0)
            ),
        }
        rows.append(row)
        # Save row-level predictions for forensic comparison.
        development.assign(pred_weekend_v2=crossfit).to_parquet(
            root / "ensemble" / f"{item['name']}_development_oof.parquet", index=False
        )
        benchmark.assign(pred_weekend_v2=benchmark_prediction).to_parquet(
            root / "ensemble" / f"{item['name']}_benchmark_oof.parquet", index=False
        )

    accepted = sorted(
        (row for row in rows if row["accepted"]),
        key=lambda row: row["selection_score"], reverse=True,
    )
    if accepted:
        winner = accepted[0]
        promote = True
    else:
        winner = {
            "name": "control",
            "plan": {"method": "control", "member": control_column},
            "development": evaluate_prediction(
                development, development[control_column].to_numpy(), control_column
            ),
            "benchmark": evaluate_prediction(
                benchmark, benchmark[control_column].to_numpy(), control_column
            ),
            "accepted": True,
            "selection_score": 0.0,
        }
        promote = False

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
        "final_submission_command": (
            "caffeinate -dimsu uv run python ml/run_weekend_v2_final.py "
            f"--recommendation {root / 'recommendation.json'} "
            f"--output-dir {root / 'final'} --device mps --resume-members"
        ),
    }
    write_json(root / "recommendation.json", recommendation)
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
    candidates = generate_weekend_v2_candidates(
        profile, seed=args.seed, prior_root=prior_root
    )
    write_json(root / "candidate_pool.json", candidates)
    write_json(root / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "profile": profile.__dict__,
        "arguments": vars(args),
        "prior_root": str(prior_root),
        "candidate_count": len(candidates),
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
            root, "screen", candidates,
            dev_origins=development_origins(profile.screen_development_origins),
            bench_origins=benchmark_origins(train, profile.screen_benchmark_origins, selected_forecasting_config()),
            epochs=profile.screen_epochs, seeds=profile.screen_seeds,
            args=args, started=started,
        )
        if args.stage == "screen" or exhausted(started, args.max_hours):
            return
    else:
        screen_results = load_stage_results(root / "screen")

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
            root, "refine", refine_candidates,
            dev_origins=development_origins(profile.refine_development_origins),
            bench_origins=benchmark_origins(train, profile.refine_benchmark_origins, selected_forecasting_config()),
            epochs=profile.refine_epochs, seeds=profile.refine_seeds,
            args=args, started=started,
        )
        if args.stage == "refine" or exhausted(started, args.max_hours):
            return
    else:
        refine_results = load_stage_results(root / "refine")

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
            root, "confirmation", confirmation_candidates,
            dev_origins=development_origins(profile.confirmation_development_origins),
            bench_origins=benchmark_origins(train, profile.confirmation_benchmark_origins, selected_forecasting_config()),
            epochs=profile.confirmation_epochs, seeds=profile.confirmation_seeds,
            args=args, started=started,
        )
        if args.stage == "confirmation" or exhausted(started, args.max_hours):
            return
    else:
        confirmation_results = load_stage_results(root / "confirmation")

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
