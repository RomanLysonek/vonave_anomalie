"""Deterministic builders for the published anomaly research snapshot."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ml.artifact_provenance import (
    PROVENANCE_SCHEMA_VERSION,
    RESULT_BODY_SCHEMA_VERSION,
    config_hash,
    dependency_manifest_hash,
    file_fingerprint,
    file_hash,
    relevant_source_hash,
    validate_artifact_manifest,
)


SCHEMA_VERSION = "anomaly-dashboard-v2"
PRODUCT_SCHEMA_VERSION = "anomaly-product-v2"
AUDIT_FILES = (
    "anomaly_metadata.json",
    "demand_anomaly_profile.csv",
    "test_context_risk.csv",
    "test_context_risk_daily.csv",
)
AUDIT_REQUIRED_COLUMNS = {
    "ProductId",
    "DateKey",
    "Quantity",
    "expected_quantity",
    "anomaly_signed_residual",
    "anomaly_score",
    "anomaly_flag",
    "anomaly_rate_28",
    "days_since_anomaly",
    "known_event",
    "systemic_anomaly_score",
    "systemic_anomaly_flag",
    "systemic_anomaly_rate_28",
    "anomaly_weight",
}
V2_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).resolve().parents[1] / "ml" / "artifact_provenance.py",
    Path(__file__).resolve().parents[1] / "ml" / "systemic_autoencoder_v2.py",
)
V2_CANONICAL_OUTPUTS = ("metadata.json", "profile.parquet")
V2_CANONICAL_INPUTS = ("data/train_data.parquet",)
V2_PROFILE_COLUMNS = {
    "DateKey",
    "autoencoder_score",
    "autoencoder_split",
}


def _clean(value: Any) -> Any:
    """Convert scientific Python values to strict JSON-compatible values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Required JSON is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Required JSON must contain an object: {path}")
    return payload


def _read_csv(path: Path, required: Iterable[str] = ()) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Required CSV is unreadable: {path}: {exc}") from exc
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _source_records(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths):
        fingerprint = file_fingerprint(path)
        if fingerprint is None:
            raise FileNotFoundError(path)
        records.append({"path": path.relative_to(root).as_posix(), **fingerprint})
    return records


def _tree_record(root: Path, path: Path) -> dict[str, Any]:
    files = [
        {"path": item.relative_to(root).as_posix(), "sha256": file_hash(item)}
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": config_hash(files),
        "files": files,
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _validate_weight_ablation(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    ablation = payload.get("forecast_weight_ablation")
    if not isinstance(ablation, dict):
        return None, "forecast_weight_ablation is missing"
    rows = ablation.get("summary")
    if not isinstance(rows, list) or not rows:
        return None, "forecast_weight_ablation.summary is empty"
    scores: dict[tuple[str, str], float] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None, "forecast_weight_ablation.summary contains a non-object"
        split, policy = row.get("split"), row.get("policy")
        wape, count = _finite_number(row.get("WAPE")), row.get("n")
        if (
            split not in {"development", "benchmark"}
            or not isinstance(policy, str)
            or not policy
            or wape is None
            or wape < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or (split, policy) in scores
        ):
            return None, "forecast_weight_ablation.summary violates its row invariants"
        scores[(split, policy)] = wape
    policies = {policy for split, policy in scores if split == "development"}
    if (
        "control" not in policies
        or policies != {policy for split, policy in scores if split == "benchmark"}
    ):
        return None, "forecast_weight_ablation must compare the same policies on both splits"
    derived_winner = min(policies, key=lambda policy: (scores[("development", policy)], policy))
    if ablation.get("winner") != derived_winner:
        return None, "forecast_weight_ablation winner contradicts validated development WAPE"
    return {
        "winner": derived_winner,
        "development_WAPE": scores[("development", derived_winner)],
        "benchmark_WAPE": scores[("benchmark", derived_winner)],
    }, "valid"


def _validate_preflight(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    source = payload.get("source")
    control = payload.get("control")
    blend = payload.get("global_convex_crossfit")
    members = payload.get("members")
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(control, dict)
        or not isinstance(blend, dict)
        or not isinstance(members, dict)
        or not members
        or any(not isinstance(value, str) or not value for value in members.values())
    ):
        return None, "preflight source, members, control or global blend is missing"
    control_dev = _finite_number(control.get("development_WAPE"))
    control_benchmark = _finite_number(control.get("benchmark_WAPE"))
    blend_dev = _finite_number(blend.get("development_WAPE"))
    blend_benchmark = _finite_number(blend.get("benchmark_WAPE"))
    probability = _finite_number(blend.get("bootstrap_probability_positive"))
    interval = blend.get("bootstrap_ci_95")
    if (
        any(value is None or value <= 0 for value in (
            control_dev, control_benchmark, blend_dev, blend_benchmark
        ))
        or probability is None
        or not 0 <= probability <= 1
        or not isinstance(interval, list)
        or len(interval) != 2
    ):
        return None, "preflight metric schema is invalid"
    low, high = (_finite_number(interval[0]), _finite_number(interval[1]))
    if low is None or high is None or low > high:
        return None, "preflight bootstrap interval is invalid"
    development_improvement = (control_dev - blend_dev) / control_dev
    benchmark_improvement = (control_benchmark - blend_benchmark) / control_benchmark
    stated_dev = _finite_number(blend.get("development_relative_improvement"))
    stated_benchmark = _finite_number(blend.get("benchmark_relative_improvement"))
    if (
        stated_dev is None
        or stated_benchmark is None
        or not math.isclose(stated_dev, development_improvement, rel_tol=1e-10, abs_tol=1e-12)
        or not math.isclose(
            stated_benchmark, benchmark_improvement, rel_tol=1e-10, abs_tol=1e-12
        )
    ):
        return None, "preflight relative improvements contradict validated WAPE values"
    return {
        "source": source,
        "development_relative_improvement": development_improvement,
        "benchmark_relative_improvement": benchmark_improvement,
        "bootstrap_probability_positive": probability,
        "bootstrap_ci_95": [low, high],
        "confidence_interval_crosses_zero": low <= 0 <= high,
    }, "valid"


def _normalise_evt_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(metadata))
    for name in ("local_evt", "systemic_evt"):
        evt = result.get(name)
        if not isinstance(evt, dict):
            continue
        if "validation_far" in evt:
            evt["validation_exceedance_rate"] = evt.pop("validation_far")
        if "validation_far_error" in evt:
            evt["validation_exceedance_error"] = evt.pop("validation_far_error")
    return result


def _snapshot_timestamp(
    profile: pd.DataFrame, context: pd.DataFrame
) -> tuple[str | None, str, str]:
    dates = pd.concat(
        [
            pd.to_datetime(profile["DateKey"], errors="coerce"),
            pd.to_datetime(context.get("DateKey", pd.Series(dtype=str)), errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if dates.empty or pd.isna(dates.max()):
        raise ValueError("Canonical dashboard inputs contain no valid snapshot dates")
    latest = dates.max()
    source_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_epoch:
        instant = datetime.fromtimestamp(int(source_epoch), tz=timezone.utc)
        return (
            instant.isoformat().replace("+00:00", "Z"),
            "SOURCE_DATE_EPOCH",
            latest.date().isoformat(),
        )
    return None, "unavailable", latest.date().isoformat()


def _audit_payload(
    profile: pd.DataFrame,
    context: pd.DataFrame,
    context_daily: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    profile = profile.copy()
    profile["DateKey"] = pd.to_datetime(profile["DateKey"], errors="coerce")
    if profile["DateKey"].isna().any():
        raise ValueError("demand_anomaly_profile.csv contains invalid DateKey values")
    for column in ("anomaly_flag", "known_event", "systemic_anomaly_flag"):
        profile[column] = profile[column].astype("boolean").fillna(False).astype(bool)
    products = sorted(int(value) for value in profile["ProductId"].unique())
    if products != list(range(1, 31)):
        raise ValueError(f"Canonical audit must contain ProductId 1..30, found {products}")

    product_summary = (
        profile.groupby("ProductId", as_index=False)
        .agg(
            observed_days=("Quantity", "count"),
            threshold_exceedances=("anomaly_flag", "sum"),
            known_event_days=("known_event", "sum"),
            max_anomaly_score=("anomaly_score", "max"),
            mean_training_weight=("anomaly_weight", "mean"),
        )
        .sort_values("ProductId")
    )
    product_summary["validation_alert_rate"] = (
        product_summary["threshold_exceedances"] / product_summary["observed_days"]
    )

    daily = (
        profile.groupby("DateKey", as_index=False)
        .agg(
            total_quantity=("Quantity", "sum"),
            threshold_exceedances=("anomaly_flag", "sum"),
            known_event_exceedances=(
                "anomaly_flag",
                lambda flags: int(
                    (
                        flags.astype(bool)
                        & profile.loc[flags.index, "known_event"].astype(bool)
                    ).sum()
                ),
            ),
            max_local_score=("anomaly_score", "max"),
            systemic_score=("systemic_anomaly_score", "max"),
            systemic_flag=("systemic_anomaly_flag", "max"),
        )
        .sort_values("DateKey")
    )
    daily["DateKey"] = daily["DateKey"].dt.strftime("%Y-%m-%d")

    top = profile.loc[profile["anomaly_flag"]].nlargest(25, "anomaly_score").copy()
    top["DateKey"] = top["DateKey"].dt.strftime("%Y-%m-%d")
    top_columns = [
        "ProductId",
        "DateKey",
        "Quantity",
        "expected_quantity",
        "anomaly_signed_residual",
        "anomaly_score",
        "known_event",
        "systemic_anomaly_flag",
        "anomaly_weight",
    ]

    context = context.copy()
    context["DateKey"] = pd.to_datetime(context["DateKey"], errors="coerce").dt.strftime("%Y-%m-%d")
    context_daily = context_daily.copy()
    context_daily["DateKey"] = pd.to_datetime(
        context_daily["DateKey"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return {
        "available": True,
        "metadata": _normalise_evt_metadata(metadata),
        "products": products,
        "product_summary": product_summary.to_dict(orient="records"),
        "daily": daily.to_dict(orient="records"),
        "top_exceedances": top[top_columns].to_dict(orient="records"),
        "context_daily": context_daily.to_dict(orient="records"),
        "context_rows": context.nlargest(20, "context_risk_percentile").to_dict(orient="records"),
        "known_event_exceedances": int((profile["anomaly_flag"] & profile["known_event"]).sum()),
    }


def _product_payload(
    profile: pd.DataFrame,
    context: pd.DataFrame,
    product_id: int,
    source_manifest_hash: str,
) -> dict[str, Any]:
    rows = profile.loc[
        pd.to_numeric(profile["ProductId"], errors="coerce").eq(product_id)
    ].copy()
    if rows.empty:
        return {
            "schema_version": PRODUCT_SCHEMA_VERSION,
            "available": False,
            "product_id": product_id,
            "message": f"Product {product_id} does not exist in the canonical anomaly audit.",
        }
    rows["DateKey"] = pd.to_datetime(rows["DateKey"], errors="coerce")
    rows = rows.sort_values("DateKey")
    rows["DateKey"] = rows["DateKey"].dt.strftime("%Y-%m-%d")
    for column in ("anomaly_flag", "known_event", "systemic_anomaly_flag"):
        rows[column] = rows[column].astype("boolean").fillna(False).astype(bool)
    future = context.loc[
        pd.to_numeric(context["ProductId"], errors="coerce").eq(product_id)
    ].copy()
    future["DateKey"] = pd.to_datetime(future["DateKey"], errors="coerce").dt.strftime("%Y-%m-%d")
    timeline_columns = [
        "DateKey",
        "Quantity",
        "expected_quantity",
        "anomaly_signed_residual",
        "anomaly_score",
        "anomaly_flag",
        "anomaly_rate_28",
        "days_since_anomaly",
        "known_event",
        "systemic_anomaly_score",
        "systemic_anomaly_flag",
        "systemic_anomaly_rate_28",
        "anomaly_weight",
    ]
    exceedances = rows.loc[rows["anomaly_flag"]].nlargest(20, "anomaly_score")
    return {
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "available": True,
        "product_id": product_id,
        "source_manifest_hash": source_manifest_hash,
        "summary": {
            "observed_days": int(rows["Quantity"].notna().sum()),
            "threshold_exceedances": int(rows["anomaly_flag"].sum()),
            "known_event_exceedances": int((rows["anomaly_flag"] & rows["known_event"]).sum()),
            "max_score": rows["anomaly_score"].max(),
            "mean_training_weight": rows["anomaly_weight"].mean(),
        },
        "timeline": rows[timeline_columns].to_dict(orient="records"),
        "top_exceedances": exceedances[timeline_columns].to_dict(orient="records"),
        "future_context": future.to_dict(orient="records"),
    }


def _v2_evidence(root: Path) -> dict[str, Any]:
    v2_root = root / "outputs" / "anomaly_autoencoder_v2"
    manifest_path = v2_root / "artifact_manifest.json"
    if not manifest_path.exists():
        return {
            "available": False,
            "state": "unavailable",
            "message": (
                "No checked-in fingerprinted V2 artifact is available. "
                "V2 is the canonical leakage-safe implementation; no result is inferred."
            ),
        }
    manifest = _read_json(manifest_path)
    canonical_inputs = manifest.get("canonical_inputs")
    configuration = manifest.get("configuration")
    canonical_outputs = manifest.get("canonical_outputs")
    body = manifest.get("body")
    body_fingerprint = manifest.get("result_body")
    if (
        manifest.get("schema_version") != "systemic-autoencoder-v2"
        or not isinstance(canonical_inputs, dict)
        or not canonical_inputs
        or not isinstance(configuration, dict)
        or not configuration
        or not isinstance(canonical_outputs, list)
        or not canonical_outputs
        or not isinstance(body, dict)
        or not body
    ):
        return {
            "available": False,
            "state": "invalid",
            "message": (
                "The checked-in V2 manifest does not satisfy the canonical "
                "input, source, dependency and output fingerprint contract."
            ),
        }
    if tuple(sorted(canonical_inputs)) != tuple(sorted(V2_CANONICAL_INPUTS)):
        return {
            "available": False,
            "state": "invalid",
            "message": "V2 manifest canonical input set is unsupported.",
        }
    input_hashes: dict[str, str] = {}
    for relative_name, expected in canonical_inputs.items():
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or not isinstance(expected, dict):
            return {
                "available": False,
                "state": "invalid",
                "message": f"V2 canonical input path is unsafe: {relative_name}",
            }
        actual = file_fingerprint(root / relative)
        if actual != expected:
            return {
                "available": False,
                "state": "invalid",
                "message": f"V2 canonical input fingerprint mismatch: {relative_name}",
            }
        input_hashes[relative.as_posix()] = actual["sha256"]
    expected_fingerprint = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": "systemic-autoencoder-v2",
        "semantic_hash": config_hash(configuration),
        "input_data_hashes": input_hashes,
        "source_hash": relevant_source_hash(V2_SOURCE_PATHS, repo_root=root),
        "dependency_manifest_hash": dependency_manifest_hash(root),
    }
    output_names: list[str] = []
    for value in canonical_outputs:
        if not isinstance(value, str):
            return {"available": False, "state": "invalid", "message": "V2 output path is invalid."}
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            return {
                "available": False,
                "state": "invalid",
                "message": f"V2 output path is unsafe: {value}",
            }
        output_names.append(relative.as_posix())
    if tuple(sorted(output_names)) != tuple(sorted(V2_CANONICAL_OUTPUTS)):
        return {
            "available": False,
            "state": "invalid",
            "message": "V2 manifest canonical output set is unsupported.",
        }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(output_names):
        return {
            "available": False,
            "state": "invalid",
            "message": "V2 manifest does not declare exactly its canonical outputs.",
        }
    valid, reason = validate_artifact_manifest(
        manifest,
        expected_fingerprint,
        base_dir=v2_root,
        required_outputs=output_names,
    )
    expected_body = {
        "schema_version": RESULT_BODY_SCHEMA_VERSION,
        "sha256": config_hash(body),
    }
    if not valid or body_fingerprint != expected_body:
        detail = reason if not valid else "canonical result-body digest mismatch"
        return {
            "available": False,
            "state": "invalid",
            "message": f"V2 canonical provenance validation failed: {detail}.",
        }
    required_body = {
        "schema_version",
        "status",
        "configuration_hash",
        "profile_rows",
        "split_counts",
    }
    if (
        set(body) != required_body
        or body.get("schema_version") != "systemic-autoencoder-v2-result-v1"
        or body.get("status") != "complete"
        or body.get("configuration_hash") != config_hash(configuration)
        or isinstance(body.get("profile_rows"), bool)
        or not isinstance(body.get("profile_rows"), int)
        or body["profile_rows"] <= 0
        or not isinstance(body.get("split_counts"), dict)
    ):
        return {
            "available": False,
            "state": "invalid",
            "message": "V2 canonical result body schema is invalid.",
        }
    split_counts = body["split_counts"]
    expected_splits = {"train", "validation", "calibration", "holdout"}
    if (
        set(split_counts) != expected_splits
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in split_counts.values()
        )
        or sum(split_counts.values()) != body["profile_rows"]
    ):
        return {
            "available": False,
            "state": "invalid",
            "message": "V2 canonical split counts contradict the profile row count.",
        }
    metadata = _read_json(v2_root / "metadata.json")
    if (
        metadata.get("schema_version") != "systemic-autoencoder-v2-metadata-v1"
        or metadata.get("configuration_hash") != config_hash(configuration)
        or metadata.get("profile_rows") != body["profile_rows"]
        or metadata.get("split_counts") != split_counts
    ):
        return {
            "available": False,
            "state": "invalid",
            "message": "V2 metadata contradicts the canonical result body.",
        }
    try:
        profile = pd.read_parquet(v2_root / "profile.parquet")
    except (OSError, ValueError, TypeError) as exc:
        return {
            "available": False,
            "state": "invalid",
            "message": f"V2 profile is unreadable: {exc}.",
        }
    dates = pd.to_datetime(profile.get("DateKey"), errors="coerce")
    scores = pd.to_numeric(profile.get("autoencoder_score"), errors="coerce")
    split_order = {"train": 0, "validation": 1, "calibration": 2, "holdout": 3}
    split_values = profile.get("autoencoder_split")
    split_ranks = (
        split_values.map(split_order)
        if isinstance(split_values, pd.Series)
        else pd.Series(dtype=float)
    )
    if (
        len(profile) != body["profile_rows"]
        or not V2_PROFILE_COLUMNS.issubset(profile.columns)
        or profile["autoencoder_split"].value_counts().to_dict() != split_counts
        or dates.isna().any()
        or not dates.is_monotonic_increasing
        or dates.duplicated().any()
        or scores.isna().any()
        or not np.isfinite(scores.to_numpy(dtype=float)).all()
        or (scores < 0).any()
        or split_ranks.isna().any()
        or not split_ranks.is_monotonic_increasing
    ):
        return {
            "available": False,
            "state": "invalid",
            "message": "V2 profile schema or split counts contradict the manifest.",
        }
    return {"available": True, "state": "verified", "manifest": manifest}


def build_anomaly_dashboard(root_dir: Path) -> dict[str, Any]:
    """Build the canonical aggregate from checked-in, verifiable evidence only."""
    root = Path(root_dir).resolve()
    audit_root = root / "outputs" / "anomaly_audit_real"
    source_paths = [audit_root / name for name in AUDIT_FILES]
    source_paths.append(root / "reports" / "weekend_v2_preflight.json")
    source_paths.append(root / "outputs" / "real_data_test_summary.json")
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Canonical dashboard input is missing: {path}")
    profile = _read_csv(audit_root / "demand_anomaly_profile.csv", AUDIT_REQUIRED_COLUMNS)
    context = _read_csv(
        audit_root / "test_context_risk.csv",
        {"ProductId", "DateKey", "context_risk_raw", "context_risk_percentile", "context_shift_flag"},
    )
    context_daily = _read_csv(
        audit_root / "test_context_risk_daily.csv",
        {"DateKey", "mean_context_risk", "max_context_risk", "shifted_products"},
    )
    metadata = _read_json(audit_root / "anomaly_metadata.json")
    preflight = _read_json(root / "reports" / "weekend_v2_preflight.json")
    verified_ablation = _read_json(root / "outputs" / "real_data_test_summary.json")
    ablation_evidence, ablation_reason = _validate_weight_ablation(verified_ablation)
    records = _source_records(root, source_paths)
    source_manifest_hash = config_hash(records)
    generated_at, timestamp_basis, snapshot_as_of = _snapshot_timestamp(profile, context)

    excluded = []
    for name in ("anomaly_autoencoder_real", "anomaly_autoencoder_recent"):
        path = root / "outputs" / name
        if path.is_dir():
            excluded.append({
                **_tree_record(root, path),
                "status": "excluded",
                "reason": (
                    "Legacy compact autoencoder preprocessing used medians derived from "
                    "the full timeline, including future observations."
                ),
            })

    recommendation = (
        {
            "policy": ablation_evidence["winner"],
            "anomaly_mode": "off",
            "status": "current",
            "verified_evidence": {
                "artifact": "outputs/real_data_test_summary.json",
                "weight_ablation_winner": ablation_evidence["winner"],
            },
            "reason": (
                "The verified weighting ablation selected control, and no "
                "provenance-complete checked-in evidence justifies anomaly promotion."
            ),
        }
        if ablation_evidence is not None and ablation_evidence["winner"] == "control"
        else {
            "policy": "unavailable",
            "anomaly_mode": "off",
            "status": "unavailable",
            "verified_evidence": {
                "artifact": "outputs/real_data_test_summary.json",
                "weight_ablation_winner": None,
            },
            "reason": f"Weight-ablation evidence is unavailable: {ablation_reason}.",
        }
    )
    preflight_payload = {
        "state": "contaminated",
        "scientific_status": "contaminated",
        "provenance": "unverified",
        "selection_use": "excluded",
        "source": "historical uploaded overnight confirmation OOF",
        "development_relative_improvement": None,
        "benchmark_relative_improvement": None,
        "bootstrap_probability_positive": None,
        "bootstrap_ci_95": None,
        "confidence_interval_crosses_zero": None,
        "message": (
            "Historical overnight/search OOF may contain benchmark-target contamination. "
            "The preflight is unverified and excluded from current candidate and selection evidence."
        ),
    }

    return _clean({
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "generated_at_basis": timestamp_basis,
        "snapshot_as_of": snapshot_as_of,
        "source_manifest_hash": source_manifest_hash,
        "snapshot_banner": {
            "version": SCHEMA_VERSION,
            "provenance": "checked-in, content-addressed inputs",
            "generated": generated_at,
            "snapshot_as_of": snapshot_as_of,
        },
        "research_status": {
            "validated_anomaly_detector": False,
            "truth_labels_available": False,
            "known_events_are_labels": False,
            "message": (
                "No anomaly truth labels exist. Threshold exceedances are review signals, "
                "and known events are explanatory proxies—not anomaly labels."
            ),
        },
        "recommendation": recommendation,
        "audit": _audit_payload(profile, context, context_daily, metadata),
        "autoencoder_v2": _v2_evidence(root),
        "excluded_evidence": excluded,
        "overnight": {
            "available": False,
            "state": "contaminated",
            "scientific_status": "contaminated",
            "provenance": "unverified",
            "selection_use": "excluded",
            "message": (
                "Historical overnight, diagnostic, and search artifacts may contain "
                "benchmark-target contamination and are excluded from current evidence."
            ),
        },
        "weekend_v2": {
            "available": False,
            "state": "not_run",
            "message": (
                "Weekend-v2 was not run. Legacy overnight/search inputs are contaminated, "
                "unverified, and excluded; control remains the recommendation."
            ),
        },
        "weekend_v2_preflight": preflight_payload,
        "product_artifact_template": "anomaly-products-v2/product-{product_id}.json",
        "source_inputs": records,
        "source_hash": relevant_source_hash(
            (
                Path(__file__),
                root / "ml" / "anomaly_detection.py",
                root / "ml" / "artifact_provenance.py",
            ),
            repo_root=root,
        ),
        "dependency_manifest_hash": dependency_manifest_hash(root),
    })


def build_product_payload(root_dir: Path, product_id: int) -> dict[str, Any]:
    """Build one product payload from the same canonical audit as the aggregate."""
    root = Path(root_dir).resolve()
    aggregate = build_anomaly_dashboard(root)
    audit_root = root / "outputs" / "anomaly_audit_real"
    profile = _read_csv(audit_root / "demand_anomaly_profile.csv", AUDIT_REQUIRED_COLUMNS)
    context = _read_csv(audit_root / "test_context_risk.csv")
    return _clean(
        _product_payload(profile, context, int(product_id), aggregate["source_manifest_hash"])
    )


def publish_anomaly_artifacts(root_dir: Path, destination: Path | None = None) -> dict[str, Any]:
    """Write the immutable aggregate and all 30 product artifacts."""
    root = Path(root_dir).resolve()
    destination = destination or root / "outputs" / "dashboard"
    aggregate = build_anomaly_dashboard(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".anomaly-dashboard-", dir=destination.parent))
    try:
        _write_json(stage / "anomaly-dashboard-v2.json", aggregate)
        for product_id in aggregate["audit"]["products"]:
            _write_json(
                stage / "anomaly-products-v2" / f"product-{product_id}.json",
                build_product_payload(root, int(product_id)),
            )
        product_ids = sorted(
            int(path.stem.removeprefix("product-"))
            for path in (stage / "anomaly-products-v2").glob("product-*.json")
        )
        if product_ids != list(range(1, 31)):
            raise RuntimeError(f"Expected product artifacts 1..30, found {product_ids}")
        backup = destination.with_name(f".{destination.name}-backup")
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(stage, destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return aggregate
