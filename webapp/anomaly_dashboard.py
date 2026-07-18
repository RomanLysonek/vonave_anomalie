"""Aggregation layer for the local Anomaly Lab dashboard.

The research pipeline writes several independent result families.  This module
turns those files into a compact, stable JSON contract for the browser while
keeping large product-level timelines behind a dedicated endpoint.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


AUDIT_ROOT_NAMES = ("anomaly_audit_real", "anomaly_audit")
AUTOENCODER_ROOT_NAMES = ("anomaly_autoencoder_real", "anomaly_autoencoder_recent")


def _clean(value: Any) -> Any:
    """Convert pandas/numpy/non-finite values into strict JSON values."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _first_existing(root: Path, relative_paths: Iterable[str]) -> Path | None:
    for relative in relative_paths:
        path = root / relative
        if path.exists():
            return path
    return None


def _records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    selected = frame.head(limit) if limit is not None else frame
    return _clean(selected.to_dict(orient="records"))


def _latest_mtime(paths: Iterable[Path]) -> str | None:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()


def _audit_root(outputs: Path) -> Path | None:
    return _first_existing(outputs, AUDIT_ROOT_NAMES)


def _audit_payload(outputs: Path) -> dict[str, Any]:
    root = _audit_root(outputs)
    if root is None:
        return {
            "available": False,
            "message": "Run `uv run python ml/run_anomaly_audit.py --output-dir outputs/anomaly_audit_real`.",
        }

    metadata = _read_json(root / "anomaly_metadata.json") or {}
    profile = _read_csv(root / "demand_anomaly_profile.csv")
    context = _read_csv(root / "test_context_risk.csv")
    context_daily = _read_csv(root / "test_context_risk_daily.csv")

    product_summary = pd.DataFrame()
    daily = pd.DataFrame()
    top_local = pd.DataFrame()
    event_anomalies = 0
    products: list[int] = []
    if not profile.empty:
        profile["DateKey"] = pd.to_datetime(profile["DateKey"], errors="coerce")
        profile["anomaly_flag"] = profile.get("anomaly_flag", False).fillna(False).astype(bool)
        profile["known_event"] = profile.get("known_event", False).fillna(False).astype(bool)
        profile["systemic_anomaly_flag"] = profile.get(
            "systemic_anomaly_flag", False
        ).fillna(False).astype(bool)
        profile["anomaly_score"] = pd.to_numeric(profile.get("anomaly_score"), errors="coerce")
        profile["Quantity"] = pd.to_numeric(profile.get("Quantity"), errors="coerce")
        products = sorted(int(value) for value in profile["ProductId"].dropna().unique())
        event_anomalies = int((profile["anomaly_flag"] & profile["known_event"]).sum())

        product_summary = (
            profile.groupby("ProductId", as_index=False)
            .agg(
                observed_days=("Quantity", "count"),
                local_anomalies=("anomaly_flag", "sum"),
                known_event_days=("known_event", "sum"),
                max_anomaly_score=("anomaly_score", "max"),
                mean_anomaly_weight=("anomaly_weight", "mean"),
            )
        )
        product_summary["local_rate"] = (
            product_summary["local_anomalies"]
            / product_summary["observed_days"].replace(0, np.nan)
        )
        product_summary = product_summary.sort_values(
            ["local_anomalies", "max_anomaly_score"], ascending=[False, False]
        )

        daily = (
            profile.dropna(subset=["DateKey"])
            .groupby("DateKey", as_index=False)
            .agg(
                total_quantity=("Quantity", "sum"),
                local_anomalies=("anomaly_flag", "sum"),
                event_anomalies=(
                    "anomaly_flag",
                    lambda values: int(
                        (
                            values.astype(bool)
                            & profile.loc[values.index, "known_event"].astype(bool)
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

        top_local = profile.loc[profile["anomaly_flag"]].sort_values(
            "anomaly_score", ascending=False
        )[[
            "ProductId", "DateKey", "Quantity", "expected_quantity",
            "anomaly_signed_residual", "anomaly_score", "known_event",
            "systemic_anomaly_flag", "anomaly_weight",
        ]].copy()
        top_local["DateKey"] = top_local["DateKey"].dt.strftime("%Y-%m-%d")

    if not context_daily.empty:
        context_daily["DateKey"] = pd.to_datetime(
            context_daily["DateKey"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    return _clean({
        "available": True,
        "root": str(root),
        "updated_at": _latest_mtime(root.glob("*")),
        "metadata": metadata,
        "event_protected_anomalies": event_anomalies,
        "products": products,
        "product_summary": _records(product_summary, 30),
        "daily": _records(daily),
        "top_local": _records(top_local, 25),
        "context_daily": _records(context_daily),
        "context_rows": _records(context.sort_values(
            "context_risk_percentile", ascending=False
        ) if not context.empty else context, 20),
    })


def _autoencoder_run_payload(root: Path) -> dict[str, Any]:
    metadata = _read_json(root / "metadata.json") or {}
    interpretation = _read_json(root / "interpretation.json") or {}
    scores_path = root / "scores_with_business_context.csv"
    if not scores_path.exists():
        scores_path = root / "systemic_autoencoder_scores.csv"
    scores = _read_csv(scores_path)
    if not scores.empty:
        scores["DateKey"] = pd.to_datetime(scores["DateKey"], errors="coerce")
        scores = scores.sort_values("DateKey")
        scores["DateKey"] = scores["DateKey"].dt.strftime("%Y-%m-%d")
    flagged = scores.loc[
        scores.get("systemic_autoencoder_flag", False).fillna(False).astype(bool)
    ] if not scores.empty and "systemic_autoencoder_flag" in scores else pd.DataFrame()
    if not flagged.empty:
        flagged = flagged.sort_values("systemic_autoencoder_score", ascending=False)
    return _clean({
        "name": root.name,
        "available": bool(metadata or not scores.empty),
        "updated_at": _latest_mtime(root.glob("*")),
        "metadata": metadata,
        "interpretation": interpretation,
        "timeline": _records(scores),
        "top_flagged": _records(flagged, 20),
    })


def _autoencoder_payload(outputs: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in AUTOENCODER_ROOT_NAMES:
        root = outputs / name
        if root.exists():
            runs.append(_autoencoder_run_payload(root))
            seen.add(root.name)
    for root in sorted(outputs.glob("anomaly_autoencoder*")):
        if root.is_dir() and root.name not in seen and (root / "metadata.json").exists():
            runs.append(_autoencoder_run_payload(root))
    return {
        "available": bool(runs),
        "runs": runs,
        "message": None if runs else "No trained autoencoder outputs were found.",
    }


def _top_rows(path: Path, *, limit: int = 8, sort_by: str | None = None,
              ascending: bool = True) -> list[dict[str, Any]]:
    frame = _read_csv(path)
    if frame.empty:
        return []
    if sort_by and sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=ascending)
    return _records(frame, limit)


def _count_trial_files(root: Path, stage: str, filename: str) -> int:
    stage_root = root / stage
    if not stage_root.exists():
        return 0
    return sum(1 for path in stage_root.glob(f"*/{filename}") if path.is_file())


def _stage_expected(root: Path, stage: str) -> int | None:
    if stage == "screen":
        manifest = _read_json(root / "manifest.json") or {}
        value = manifest.get("candidate_count")
        return int(value) if value is not None else None
    payload = _read_json(root / f"{stage}_candidates.json")
    return len(payload) if isinstance(payload, list) else None


def _weekend_search_payload(outputs: Path, name: str) -> dict[str, Any]:
    root = outputs / name
    if not root.exists():
        return {"available": False, "name": name, "state": "not_started"}
    recommendation = _read_json(root / "recommendation.json") or {}
    stages = {}
    for stage in ("screen", "refine", "confirmation"):
        stages[stage] = {
            "completed": _count_trial_files(root, stage, "result.json"),
            "failed": _count_trial_files(root, stage, "failure.json"),
            "expected": _stage_expected(root, stage),
        }
    any_completed = any(item["completed"] for item in stages.values())
    state = "complete" if recommendation else ("running" if any_completed or (root / "manifest.json").exists() else "not_started")
    winner = recommendation.get("winner") if isinstance(recommendation, dict) else None
    comparisons = recommendation.get("comparisons", []) if isinstance(recommendation, dict) else []
    comparisons = sorted(
        comparisons,
        key=lambda row: float(row.get("selection_score", -999.0)),
        reverse=True,
    )[:10]
    return _clean({
        "available": True,
        "name": name,
        "state": state,
        "updated_at": _latest_mtime(root.rglob("*")),
        "manifest": _read_json(root / "manifest.json") or {},
        "stages": stages,
        "failed_total": sum(item["failed"] for item in stages.values()),
        "screen_leaderboard": _top_rows(root / "screen_leaderboard.csv", limit=8),
        "refine_leaderboard": _top_rows(root / "refine_leaderboard.csv", limit=8),
        "confirmation_leaderboard": _top_rows(root / "confirmation_leaderboard.csv", limit=8),
        "ensemble_leaderboard": _top_rows(
            root / "ensemble_leaderboard.csv", limit=12,
            sort_by="selection_score", ascending=False,
        ),
        "recommendation": {
            "promote_weekend_v2": recommendation.get("promote_weekend_v2"),
            "winner": winner,
            "final_submission_command": recommendation.get("final_submission_command"),
            "comparisons": comparisons,
        } if recommendation else None,
    })


def _overnight_payload(outputs: Path) -> dict[str, Any]:
    root = outputs / "overnight_anomaly_search"
    if not root.exists():
        return {"available": False, "message": "No completed overnight search was found."}
    recommendation = _read_json(root / "recommendation.json") or {}
    failure_count = sum(1 for path in root.rglob("failure.json") if path.is_file())
    diagnostic = _read_csv(root / "diagnostic_leaderboard.csv")
    proxy = _read_csv(root / "proxy_leaderboard.csv")
    neural = _read_csv(root / "neural_leaderboard.csv")
    confirmation = _read_csv(root / "confirmation_leaderboard.csv")
    return _clean({
        "available": True,
        "updated_at": _latest_mtime(root.rglob("*")),
        "counts": {
            "diagnostic": len(diagnostic),
            "proxy": len(proxy),
            "neural": len(neural),
            "confirmation": len(confirmation),
            "failed": failure_count,
        },
        "diagnostic_leaderboard": _records(diagnostic, 8),
        "proxy_leaderboard": _records(proxy, 8),
        "neural_leaderboard": _records(neural, 8),
        "confirmation_leaderboard": _records(confirmation, 8),
        "recommendation": {
            "winner": recommendation.get("winner"),
            "promote_anomaly_layer": recommendation.get("promote_anomaly_layer"),
            "final_submission_command": recommendation.get("final_submission_command"),
            "comparisons": recommendation.get("comparisons", []),
        } if recommendation else None,
    })


def build_anomaly_dashboard(root_dir: Path) -> dict[str, Any]:
    outputs = root_dir / "outputs"
    preflight = _read_json(root_dir / "reports" / "weekend_v2_preflight.json")
    payload = {
        "schema_version": "anomaly-dashboard-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit": _audit_payload(outputs),
        "autoencoder": _autoencoder_payload(outputs),
        "overnight": _overnight_payload(outputs),
        "weekend_v2": _weekend_search_payload(outputs, "weekend_v2_search"),
        "weekend_v2_smoke": _weekend_search_payload(outputs, "weekend_v2_smoke"),
        "weekend_v2_preflight": preflight,
    }
    return _clean(payload)


def build_product_payload(root_dir: Path, product_id: int) -> dict[str, Any]:
    outputs = root_dir / "outputs"
    audit_root = _audit_root(outputs)
    if audit_root is None:
        return {
            "available": False,
            "product_id": product_id,
            "message": "No anomaly audit output is available.",
        }
    profile = _read_csv(audit_root / "demand_anomaly_profile.csv")
    if profile.empty:
        return {
            "available": False,
            "product_id": product_id,
            "message": "The anomaly profile is empty.",
        }
    profile = profile.loc[pd.to_numeric(profile["ProductId"], errors="coerce") == product_id].copy()
    if profile.empty:
        return {
            "available": False,
            "product_id": product_id,
            "message": f"Product {product_id} does not exist in the anomaly profile.",
        }
    profile["DateKey"] = pd.to_datetime(profile["DateKey"], errors="coerce")
    profile = profile.sort_values("DateKey")
    profile["DateKey"] = profile["DateKey"].dt.strftime("%Y-%m-%d")
    columns = [
        "DateKey", "Quantity", "expected_quantity", "anomaly_signed_residual",
        "anomaly_score", "anomaly_flag", "anomaly_rate_28", "days_since_anomaly",
        "known_event", "systemic_anomaly_score", "systemic_anomaly_flag",
        "systemic_anomaly_rate_28", "anomaly_weight",
    ]
    timeline = profile[[column for column in columns if column in profile.columns]]
    anomalies = profile.loc[profile.get("anomaly_flag", False).fillna(False).astype(bool)]
    anomalies = anomalies.sort_values("anomaly_score", ascending=False)

    context_path = audit_root / "test_context_risk.csv"
    context = _read_csv(context_path)
    if not context.empty:
        context = context.loc[pd.to_numeric(context["ProductId"], errors="coerce") == product_id].copy()
        context["DateKey"] = pd.to_datetime(context["DateKey"], errors="coerce").dt.strftime("%Y-%m-%d")

    return _clean({
        "available": True,
        "product_id": product_id,
        "summary": {
            "observed_days": int(pd.to_numeric(profile.get("Quantity"), errors="coerce").notna().sum()),
            "local_anomalies": int(profile.get("anomaly_flag", False).fillna(False).astype(bool).sum()),
            "event_anomalies": int((
                profile.get("anomaly_flag", False).fillna(False).astype(bool)
                & profile.get("known_event", False).fillna(False).astype(bool)
            ).sum()),
            "max_score": pd.to_numeric(profile.get("anomaly_score"), errors="coerce").max(),
            "mean_weight": pd.to_numeric(profile.get("anomaly_weight"), errors="coerce").mean(),
        },
        "timeline": _records(timeline),
        "top_anomalies": _records(anomalies[columns], 20),
        "future_context": _records(context),
    })


def build_anomaly_status(root_dir: Path) -> dict[str, Any]:
    """Lightweight live-search payload used by the browser's periodic refresh."""
    outputs = root_dir / "outputs"
    audit_root = _audit_root(outputs)
    audit_metadata = (
        _read_json(audit_root / "anomaly_metadata.json") if audit_root is not None else None
    ) or {}
    return _clean({
        "schema_version": "anomaly-dashboard-status-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_summary": {
            "n_local_anomalies": audit_metadata.get("n_local_anomalies"),
            "n_systemic_days": audit_metadata.get("n_systemic_days"),
        },
        "overnight": _overnight_payload(outputs),
        "weekend_v2": _weekend_search_payload(outputs, "weekend_v2_search"),
        "weekend_v2_smoke": _weekend_search_payload(outputs, "weekend_v2_smoke"),
    })
