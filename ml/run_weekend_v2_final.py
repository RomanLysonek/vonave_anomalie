"""Train the weekend-v2 experts on all history and create the final blend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from anomaly_search_common import apply_candidate_config, load_json, selected_forecasting_config, write_json
from framework import Config, compute_baseline, load_raw
from models.naive_baselines import moving_average_predict, seasonal_naive_predict
from pipeline import run_final_forecast_direct
from weekend_v2_common import apply_weight_plan
from artifact_provenance import (
    artifact_fingerprint,
    config_hash,
    dataframe_content_hash,
    environment_metadata,
    file_fingerprint,
    FINAL_MEMBER_SOURCE_PATHS,
    output_fingerprints,
    neural_training_identity,
    resolve_compute_device,
    validate_artifact_manifest,
    load_validated_result,
)


MEMBER_CACHE_SCHEMA_VERSION = "weekend-v2-final-member-v5"
RECOMMENDATION_MANIFEST_SCHEMA_VERSION = "weekend-v2-recommendation-provenance-v3"
MEMBER_KEY_COLUMNS = ("ProductId", "DateKey")
SAFE_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TRUSTED_PLAN_METHODS = {
    "control",
    "global_convex",
    "horizon_convex",
    "product_convex",
    "aggregate_reconciled",
}


def _member_fingerprint(
    candidate: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    epochs: int,
    seeds: tuple[int, ...],
    device: str,
) -> dict:
    return artifact_fingerprint(
        schema_version=MEMBER_CACHE_SCHEMA_VERSION,
        semantic={
            "candidate": candidate,
            "epochs": epochs,
            "seeds": list(seeds),
            "requested_device": device,
            "resolved_device": resolve_compute_device(device),
            "neural_training_identity": neural_training_identity(
                apply_candidate_config(selected_forecasting_config(), candidate)
            ),
        },
        dataframes={
            "train": train,
            "test": test,
        },
        source_paths=FINAL_MEMBER_SOURCE_PATHS,
    )


def _normalized_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[list(MEMBER_KEY_COLUMNS)].copy()
    keys["ProductId"] = pd.to_numeric(keys["ProductId"], errors="raise")
    keys["DateKey"] = pd.to_datetime(keys["DateKey"], errors="raise")
    return keys.reset_index(drop=True)


def _validate_neural_execution(
    execution: object,
    expected_identity: dict,
    seeds: tuple[int, ...],
) -> list[dict]:
    if not isinstance(execution, list) or len(execution) != len(seeds):
        raise ValueError("member cache neural execution rows do not match seeds")
    allowed = {expected_identity["resolved_backend"]}
    if expected_identity["oom_fallback_policy"] == "device_resident_to_dataloader_on_oom":
        allowed.add("dataloader_fallback")
    for row, seed in zip(execution, seeds, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("seed") != int(seed)
            or row.get("device") != expected_identity["device"]
            or row.get("backend") not in allowed
            or row.get("batch_size") != expected_identity["batch_size"]
            or row.get("reference_batch_size")
            != expected_identity["reference_batch_size"]
        ):
            raise ValueError("member cache neural execution identity mismatch")
    return execution


def _load_resumable_member(
    member_dir: Path,
    expected_fingerprint: dict,
    test_keys: pd.DataFrame,
    expected_neural_identity: dict,
    seeds: tuple[int, ...],
    *,
    confirm_recompute_stale: bool = False,
) -> pd.DataFrame | None:
    predictions_path = member_dir / "predictions.csv"
    manifest_path = member_dir / "predictions.manifest.json"
    if not predictions_path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = load_json(manifest_path)
        valid, reason = validate_artifact_manifest(
            manifest,
            expected_fingerprint,
            base_dir=member_dir,
            required_outputs=(predictions_path.name,),
        )
        if not valid:
            raise ValueError(reason)
        execution_record = manifest.get("neural_execution")
        if not isinstance(execution_record, dict):
            raise ValueError("member cache neural execution manifest is missing")
        execution = _validate_neural_execution(
            execution_record.get("rows"), expected_neural_identity, seeds
        )
        if execution_record.get("sha256") != config_hash(execution):
            raise ValueError("member cache neural execution digest mismatch")
        cached = pd.read_csv(predictions_path)
        required = {*MEMBER_KEY_COLUMNS, "prediction"}
        if not required.issubset(cached.columns):
            raise ValueError("predictions CSV schema is missing required columns")
        if manifest.get("csv_schema") != {
            "columns": list(cached.columns),
            "key_columns": list(MEMBER_KEY_COLUMNS),
            "rows": len(cached),
        }:
            raise ValueError("predictions CSV schema mismatch")
        if dataframe_content_hash(_normalized_keys(cached)) != dataframe_content_hash(
            _normalized_keys(test_keys)
        ):
            raise ValueError("predictions CSV keys do not match test data")
        if not np.isfinite(
            pd.to_numeric(cached["prediction"], errors="coerce").to_numpy(dtype=float)
        ).all():
            raise ValueError("predictions CSV contains invalid predictions")
        return cached
    except Exception as exc:
        if not confirm_recompute_stale:
            raise RuntimeError(
                f"Stale or unverifiable member cache {predictions_path}: {exc}. "
                "Pass --confirm-recompute-stale to deliberately retrain it."
            ) from exc
        print(f"[resume] confirmed recompute of invalid member cache {predictions_path}: {exc}")
        return None


def _save_member_cache(
    member_dir: Path,
    cached_frame: pd.DataFrame,
    fingerprint: dict,
    neural_execution: list[dict],
) -> None:
    member_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = member_dir / "predictions.csv"
    tmp_path = predictions_path.with_suffix(".tmp.csv")
    cached_frame.to_csv(tmp_path, index=False)
    tmp_path.replace(predictions_path)
    write_json(
        member_dir / "predictions.manifest.json",
        {
            "schema_version": MEMBER_CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "outputs": output_fingerprints(member_dir, (predictions_path.name,)),
            "csv_schema": {
                "columns": list(cached_frame.columns),
                "key_columns": list(MEMBER_KEY_COLUMNS),
                "rows": len(cached_frame),
            },
            "neural_execution": {
                "schema_version": "weekend-v2-neural-execution-v1",
                "rows": neural_execution,
                "sha256": config_hash(neural_execution),
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendation", required=True)
    parser.add_argument("--output-dir", default="outputs/weekend_v2_search/final")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", default="42,123,777")
    parser.add_argument("--resume-members", action="store_true")
    parser.add_argument("--confirm-recompute-stale", action="store_true")
    return parser.parse_args()


def _safe_relative_path(base: Path, raw: str) -> Path:
    relative = Path(raw)
    if not raw or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe provenance path: {raw}")
    resolved_base = base.resolve()
    resolved = (resolved_base / relative).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise RuntimeError(f"Unsafe provenance path: {raw}") from exc
    return resolved


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Declarative plan {label} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise RuntimeError(f"Declarative plan {label} must be finite")
    return number


def _validate_weights(value: object, columns: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != columns:
        raise RuntimeError("Recommendation weights do not match authenticated members")
    weights = [_finite_number(value[column], f"{label}.{column}") for column in columns]
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise RuntimeError(f"Declarative plan {label} weights must be in [0, 1]")
    if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-8):
        raise RuntimeError(f"Declarative plan {label} weights must sum to one")


def _validate_indexed_weights(
    value: object,
    columns: set[str],
    label: str,
    *,
    maximum_key: int | None = None,
) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"Declarative plan {label} must be an object")
    for key, weights in value.items():
        try:
            parsed = int(key)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Declarative plan {label} key is invalid: {key!r}") from exc
        if str(parsed) != key or parsed <= 0 or (
            maximum_key is not None and parsed > maximum_key
        ):
            raise RuntimeError(f"Declarative plan {label} key is out of bounds: {key!r}")
        _validate_weights(weights, columns, f"{label}.{key}")


def _validate_declarative_plan(plan: object, columns: set[str]) -> None:
    if not isinstance(plan, dict):
        raise RuntimeError("Recommendation winner plan is invalid")
    method = plan.get("method")
    if method not in TRUSTED_PLAN_METHODS or "model_path" in plan:
        raise RuntimeError(
            "Unsupported/untrusted recommendation artifact: only fixed declarative "
            "control or numeric convex plans can execute"
        )
    allowed_keys = {
        "control": {"method", "member"},
        "global_convex": {"method", "weights"},
        "horizon_convex": {
            "method", "global_weights", "horizon_weights", "shrinkage_rows"
        },
        "product_convex": {
            "method", "global_weights", "product_weights", "shrinkage_rows"
        },
        "aggregate_reconciled": {
            "method", "weights", "horizon_scales", "shrinkage_demand"
        },
    }[method]
    if set(plan) != allowed_keys:
        raise RuntimeError("Declarative plan contains unsupported fields")
    if method == "control":
        if plan["member"] not in columns:
            raise RuntimeError("Control plan references an unauthenticated member")
        return
    if method in {"global_convex", "aggregate_reconciled"}:
        _validate_weights(plan["weights"], columns, "weights")
    else:
        _validate_weights(plan["global_weights"], columns, "global_weights")
    if method == "horizon_convex":
        _validate_indexed_weights(
            plan["horizon_weights"], columns, "horizon_weights", maximum_key=7
        )
        if _finite_number(plan["shrinkage_rows"], "shrinkage_rows") <= 0:
            raise RuntimeError("Declarative plan shrinkage_rows must be positive")
    elif method == "product_convex":
        _validate_indexed_weights(plan["product_weights"], columns, "product_weights")
        if _finite_number(plan["shrinkage_rows"], "shrinkage_rows") <= 0:
            raise RuntimeError("Declarative plan shrinkage_rows must be positive")
    elif method == "aggregate_reconciled":
        scales = plan["horizon_scales"]
        if not isinstance(scales, dict):
            raise RuntimeError("Declarative plan horizon_scales must be an object")
        for key, value in scales.items():
            if (
                not isinstance(key, str)
                or not key.isdigit()
                or str(int(key)) != key
                or not 1 <= int(key) <= 7
            ):
                raise RuntimeError(f"Declarative plan horizon scale key is invalid: {key!r}")
            scale = _finite_number(value, f"horizon_scales.{key}")
            if not 0.85 <= scale <= 1.15:
                raise RuntimeError("Declarative plan horizon scale is out of bounds")
        if _finite_number(plan["shrinkage_demand"], "shrinkage_demand") <= 0:
            raise RuntimeError("Declarative plan shrinkage_demand must be positive")


def _contains_model_path(value: object) -> bool:
    if isinstance(value, dict):
        return "model_path" in value or any(
            _contains_model_path(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_model_path(item) for item in value)
    return False


def _validate_execution_contract(recommendation: dict) -> None:
    if _contains_model_path(recommendation.get("winner")):
        raise RuntimeError(
            "Unsupported/untrusted recommendation artifact: model_path is forbidden"
        )
    if (
        recommendation.get("execution_enabled") is not True
        or recommendation.get("status") != "verified_declarative"
    ):
        raise RuntimeError(
            "Unsupported/untrusted recommendation artifact: execution is not enabled"
        )


def _validate_plan_member_identity(recommendation: dict) -> None:
    members = recommendation.get("members")
    if not isinstance(members, list) or not members:
        raise RuntimeError("Recommendation member list is missing")
    columns: set[str] = set()
    candidate_ids: set[str] = set()
    for member in members:
        candidate = member.get("candidate") if isinstance(member, dict) else None
        column = member.get("column") if isinstance(member, dict) else None
        candidate_id = candidate.get("id") if isinstance(candidate, dict) else None
        if (
            not isinstance(candidate_id, str)
            or SAFE_CANDIDATE_ID.fullmatch(candidate_id) is None
            or column != f"member__{candidate_id}"
            or column in columns
            or candidate_id in candidate_ids
        ):
            raise RuntimeError("Recommendation member/candidate identity mismatch")
        columns.add(column)
        candidate_ids.add(candidate_id)
    _validate_declarative_plan(
        recommendation.get("winner", {}).get("plan"), columns
    )
    referenced = active_members(recommendation)
    if not referenced or not referenced.issubset(columns):
        raise RuntimeError("Recommendation plan references unauthenticated members")
    reference = recommendation.get("reference_member")
    if reference is not None and reference not in columns:
        raise RuntimeError("Recommendation reference member is unauthenticated")


def _validate_recommendation(
    recommendation_path: Path,
    recommendation: dict,
) -> dict:
    _validate_execution_contract(recommendation)
    manifest_name = recommendation.get("provenance_manifest")
    if not isinstance(manifest_name, str):
        raise RuntimeError(
            "Recommendation is legacy/unverifiable: no provenance manifest. "
            "Do not reuse it; a future search must deliberately produce and confirm "
            "a content-bound recommendation."
        )
    if recommendation.get("schema_version") != "weekend-v2-search-v5":
        raise RuntimeError("Recommendation schema is unsupported or unverifiable")
    manifest_path = _safe_relative_path(recommendation_path.parent, manifest_name)
    if not manifest_path.exists():
        raise RuntimeError(f"Recommendation provenance manifest is missing: {manifest_path}")
    provenance = load_json(manifest_path)
    if provenance.get("schema_version") != RECOMMENDATION_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("Recommendation provenance schema is unsupported or unverifiable")
    if provenance.get("recommendation_file") != file_fingerprint(recommendation_path):
        raise RuntimeError("Recommendation JSON hash/size does not match its provenance manifest")
    if provenance.get("recommendation_body_sha256") != config_hash(recommendation):
        raise RuntimeError("Recommendation canonical body digest mismatch")

    search_record = provenance.get("search_manifest")
    if not isinstance(search_record, dict):
        raise RuntimeError("Recommendation is missing its bound search input")
    search_path = _safe_relative_path(
        recommendation_path.parent, str(search_record.get("path", ""))
    )
    if search_record.get("file") != file_fingerprint(search_path):
        raise RuntimeError("Bound search manifest hash/size mismatch")
    search_manifest = load_json(search_path)
    if search_record.get("input_fingerprint") != search_manifest.get("artifact_fingerprint"):
        raise RuntimeError("Bound search input fingerprint mismatch")

    member_bindings = [
        {
            "column": member["column"],
            "candidate_id": member["candidate"]["id"],
            "candidate_body_sha256": config_hash(member["candidate"]),
            "source_result_path": member.get("source_result_path"),
            "expected_result_fingerprint": member.get("expected_result_fingerprint"),
            "canonical_result_body_digest": member.get(
                "canonical_result_body_digest"
            ),
            "development_summary_sha256": member.get(
                "development_summary_sha256"
            ),
            "benchmark_summary_sha256": member.get("benchmark_summary_sha256"),
            "oof_output_fingerprints": member.get("oof_output_fingerprints"),
        }
        for member in recommendation.get("members", [])
    ]
    if (
        provenance.get("member_bindings") != member_bindings
        or provenance.get("member_bindings_sha256") != config_hash(member_bindings)
    ):
        raise RuntimeError("Recommendation member identities/candidate bodies are not bound")
    _validate_plan_member_identity(recommendation)
    for member in recommendation["members"]:
        candidate_id = member["candidate"]["id"]
        expected_fingerprint = member.get("expected_result_fingerprint")
        if not isinstance(expected_fingerprint, dict):
            raise RuntimeError(f"Member {candidate_id} expected fingerprint is missing")
        result_path = _safe_relative_path(
            recommendation_path.parent, str(member.get("source_result_path", ""))
        )
        result, reason = load_validated_result(
            result_path,
            expected_fingerprint,
            required_outputs=("development_oof.parquet", "benchmark_oof.parquet"),
        )
        if result is None:
            raise RuntimeError(f"Member source result is invalid ({candidate_id}): {reason}")
        result_manifest = result["artifact_manifest"]
        if (
            result.get("candidate") != member["candidate"]
            or member.get("candidate_body_sha256") != config_hash(result["candidate"])
            or member.get("canonical_result_body_digest")
            != result_manifest.get("result_body")
            or member.get("development_summary_sha256")
            != config_hash(result.get("development"))
            or member.get("benchmark_summary_sha256")
            != config_hash(result.get("benchmark"))
        ):
            raise RuntimeError(f"Member source identity/body/summary mismatch: {candidate_id}")
        expected_oof = member.get("oof_output_fingerprints")
        actual_outputs = result_manifest.get("outputs", {})
        if (
            not isinstance(expected_oof, dict)
            or expected_oof
            != {
                name: actual_outputs.get(name)
                for name in ("development_oof.parquet", "benchmark_oof.parquet")
            }
        ):
            raise RuntimeError(f"Member source OOF fingerprint mismatch: {candidate_id}")
    winner = recommendation.get("winner")
    if (
        not isinstance(winner, dict)
        or provenance.get("winner_body_sha256") != config_hash(winner)
        or provenance.get("winner_plan_sha256") != config_hash(winner.get("plan"))
    ):
        raise RuntimeError("Recommendation winner body/weight plan digest mismatch")
    if provenance.get("artifact_requirements") != []:
        raise RuntimeError("Recommendation contains untrusted artifact requirements")
    return provenance


def active_members(recommendation: dict) -> set[str]:
    winner = recommendation["winner"]
    plan = winner["plan"]
    method = plan["method"]
    if method == "control":
        return {plan["member"]}
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
    _validate_recommendation(recommendation_path, recommendation)
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
    reused = []
    for item in recommendation["members"]:
        column = item["column"]
        if column not in needed:
            continue
        candidate = item["candidate"]
        identity_cfg = selected_forecasting_config()
        apply_candidate_config(identity_cfg, candidate)
        identity_cfg.final_epochs = args.epochs
        identity_cfg.seeds = seeds
        expected_neural_identity = neural_training_identity(identity_cfg)
        members_root = (output / "members").resolve()
        member_dir = (members_root / candidate["id"]).resolve()
        try:
            member_dir.relative_to(members_root)
        except ValueError as exc:
            raise RuntimeError("Recommendation candidate ID escapes member output root") from exc
        member_fingerprint = _member_fingerprint(
            candidate,
            train,
            test,
            epochs=args.epochs,
            seeds=seeds,
            device=args.device,
        )
        cache_exists = any(
            (member_dir / name).exists()
            for name in ("predictions.csv", "predictions.manifest.json")
        )
        if (
            cache_exists
            and not args.resume_members
            and not args.confirm_recompute_stale
        ):
            raise RuntimeError(
                f"Existing member cache {member_dir} would be recomputed. Use "
                "--resume-members to validate/reuse it or pass "
                "--confirm-recompute-stale to deliberately replace it."
            )
        cached_frame = (
            _load_resumable_member(
                member_dir,
                member_fingerprint,
                frame,
                expected_neural_identity,
                seeds,
                confirm_recompute_stale=args.confirm_recompute_stale,
            )
            if args.resume_members
            else None
        )
        if cached_frame is not None:
            predictions = cached_frame["prediction"].to_numpy(dtype=float)
            diagnostics = {
                column: cached_frame[column].to_numpy(dtype=float)
                for column in cached_frame.columns
                if column.startswith("anomaly_") or column.startswith("autoencoder_")
            }
            reused.append(candidate["id"])
        else:
            cfg = selected_forecasting_config()
            apply_candidate_config(cfg, candidate)
            cfg.final_epochs = args.epochs
            cfg.seeds = seeds
            cfg.autoencoder_device = args.device
            cfg.autoencoder_cache_dir = str(output / "autoencoder_cache")
            cfg.allow_autoencoder_cache_build = True
            cfg.confirm_recompute_stale = args.confirm_recompute_stale
            cfg.output_dir = str(member_dir)
            _, predictions, diagnostics = run_final_forecast_direct(
                train, test, cfg, return_diagnostics=True
            )
            neural_execution = _validate_neural_execution(
                diagnostics.pop("training_execution"),
                expected_neural_identity,
                seeds,
            )
            cached_frame = pd.DataFrame({
                "ProductId": frame["ProductId"],
                "DateKey": frame["DateKey"],
                "prediction": predictions,
            })
            for key, values in diagnostics.items():
                if key.startswith("anomaly_") or key.startswith("autoencoder_"):
                    cached_frame[key] = np.asarray(values, dtype=float)
            _save_member_cache(
                member_dir,
                cached_frame,
                member_fingerprint,
                neural_execution,
            )
            write_json(member_dir / "candidate.json", candidate)
            trained.append(candidate["id"])
        frame[column] = np.asarray(predictions, dtype=float)
        for key, values in diagnostics.items():
            if key == "prediction":
                continue
            if key.startswith("anomaly_") or key.startswith("autoencoder_"):
                feature_key = f"feature__{candidate['id']}__{key}"
                frame[feature_key] = np.asarray(values, dtype=float)

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
    else:
        raise RuntimeError(f"Unsupported declarative plan method: {method}")

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
        "schema_version": "weekend-v2-final-v4",
        "recommendation": str(recommendation_path),
        "winner": winner,
        "trained_member_ids": trained,
        "reused_member_ids": reused,
        "epochs": args.epochs,
        "seeds": list(seeds),
        "rows": len(frame),
        "environment": environment_metadata(requested_device=args.device),
    })
    print(json.dumps({
        "winner": winner["name"],
        "method": method,
        "submission": str(output / "submission.csv"),
        "trained_members": trained,
        "reused_members": reused,
    }, indent=2))


if __name__ == "__main__":
    main()
