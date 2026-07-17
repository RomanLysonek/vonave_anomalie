"""Deterministic provenance and integrity helpers for reusable ML artifacts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PROVENANCE_SCHEMA_VERSION = "artifact-provenance-v3"
RESULT_BODY_SCHEMA_VERSION = "canonical-result-body-v1"

ML_ROOT = Path(__file__).resolve().parent
FORECAST_TRIAL_SOURCE_PATHS = tuple(
    ML_ROOT / path
    for path in (
        "artifact_provenance.py",
        "run_anomaly_forecast_trial.py",
        "anomaly_search_common.py",
        "pipeline.py",
        "framework.py",
        "ensemble.py",
        "anomaly_detection.py",
        "systemic_autoencoder_v2.py",
        "models/neural_net.py",
        "models/dynamic_ridge.py",
        "models/naive_baselines.py",
    )
)
DIAGNOSTIC_TRIAL_SOURCE_PATHS = tuple(
    ML_ROOT / path
    for path in (
        "artifact_provenance.py",
        "run_autoencoder_diagnostic_trial.py",
        "anomaly_search_common.py",
        "framework.py",
        "anomaly_detection.py",
        "systemic_autoencoder_v2.py",
    )
)
FINAL_MEMBER_SOURCE_PATHS = tuple(
    ML_ROOT / path
    for path in (
        "artifact_provenance.py",
        "run_weekend_v2_final.py",
        "weekend_v2_common.py",
        "anomaly_search_common.py",
        "pipeline.py",
        "framework.py",
        "ensemble.py",
        "anomaly_detection.py",
        "systemic_autoencoder_v2.py",
        "models/neural_net.py",
        "models/naive_baselines.py",
    )
)
OVERNIGHT_SEARCH_SOURCE_PATHS = tuple(
    ML_ROOT / path
    for path in (
        "artifact_provenance.py",
        "run_overnight_anomaly_search.py",
        "anomaly_search_common.py",
        "framework.py",
    )
)
WEEKEND_SEARCH_SOURCE_PATHS = tuple(
    ML_ROOT / path
    for path in (
        "artifact_provenance.py",
        "run_weekend_v2_search.py",
        "weekend_v2_common.py",
        "anomaly_search_common.py",
        "framework.py",
    )
)


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=lambda item: canonical_json(item))
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": str(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Return stable, whitespace-free JSON for common scientific Python values."""
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def config_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


candidate_hash = config_hash


def canonical_result_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the persisted result body, excluding only its enclosing manifest."""
    return {
        str(key): value
        for key, value in payload.items()
        if key != "artifact_manifest" and not str(key).startswith("_")
    }


def result_body_manifest(payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "schema_version": RESULT_BODY_SCHEMA_VERSION,
        "sha256": config_hash(canonical_result_body(payload)),
    }


def file_hash(path: str | Path) -> str | None:
    """Hash file bytes, returning ``None`` when the file is unavailable."""
    target = Path(path)
    try:
        if not stat.S_ISREG(target.lstat().st_mode):
            return None
        digest = sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return None


def file_fingerprint(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    digest = file_hash(target)
    if digest is None:
        return None
    try:
        info = target.lstat()
        if not stat.S_ISREG(info.st_mode):
            return None
        size = info.st_size
    except OSError:
        return None
    return {"sha256": digest, "size": int(size)}


def dataframe_content_hash(frame: pd.DataFrame) -> str:
    """Hash DataFrame schema, index, row order, values, and missingness."""
    schema = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_names": [str(name) if name is not None else None for name in frame.index.names],
        "index_dtype": str(frame.index.dtype),
        "shape": list(frame.shape),
    }
    digest = sha256(canonical_json(schema).encode("utf-8"))
    try:
        row_hashes = pd.util.hash_pandas_object(
            frame, index=True, categorize=False
        ).to_numpy(dtype=np.uint64, copy=False)
    except (TypeError, ValueError):
        normalized = frame.map(
            lambda value: canonical_json(value)
            if isinstance(value, (dict, list, tuple, set))
            else value
        )
        row_hashes = pd.util.hash_pandas_object(
            normalized, index=True, categorize=False
        ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(row_hashes.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _repository_root(start: str | Path | None = None) -> Path:
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return current


def relevant_source_hash(
    paths: Iterable[str | Path],
    *,
    repo_root: str | Path | None = None,
) -> str:
    root = Path(repo_root).resolve() if repo_root else _repository_root()
    records = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        candidates = (
            sorted(
                item
                for item in path.rglob("*.py")
                if item.is_file() and "__pycache__" not in item.parts
            )
            if path.is_dir()
            else [path]
        )
        for candidate in candidates:
            try:
                name = candidate.relative_to(root).as_posix()
            except ValueError:
                name = candidate.name
            records.append({"path": name, "sha256": file_hash(candidate)})
    return config_hash(sorted(records, key=lambda item: item["path"]))


def dependency_manifest_hash(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root).resolve() if repo_root else _repository_root()
    records = [
        {"path": name, "sha256": file_hash(root / name)}
        for name in ("pyproject.toml", "uv.lock")
        if (root / name).exists()
    ]
    return config_hash(records)


def git_metadata(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Observe Git metadata when available; never infer or fabricate it."""
    root = Path(repo_root).resolve() if repo_root else _repository_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip() or None
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def environment_metadata(
    *,
    requested_device: str | None = None,
    resolved_device: str | None = None,
    packages: Iterable[str] = ("numpy", "pandas", "torch"),
) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "git": git_metadata(),
    }


def resolve_compute_device(requested: str) -> str:
    """Resolve a cache-relevant torch device without guessing availability."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def neural_training_identity(
    cfg: Any, *, device: str | None = None
) -> dict[str, Any]:
    """Return the actual forecasting-device and nominal backend contract."""
    from models.neural_net import DEVICE, resolve_training_backend

    return {
        "device": device or DEVICE.type,
        "requested_backend": cfg.nn_training_backend,
        "resolved_backend": resolve_training_backend(cfg),
        "batch_size": int(cfg.batch_size),
        "reference_batch_size": int(cfg.reference_batch_size),
        "batch_semantics": "rows_per_optimizer_step_with_singleton_tail_rebalanced",
        "oom_fallback_policy": (
            "device_resident_to_dataloader_on_oom"
            if cfg.nn_training_backend == "auto"
            and resolve_training_backend(cfg) == "device_resident"
            else "disabled"
        ),
    }


def artifact_fingerprint(
    *,
    schema_version: str,
    semantic: Any,
    dataframes: Mapping[str, pd.DataFrame] | None = None,
    source_paths: Iterable[str | Path] = (),
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root else _repository_root()
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": schema_version,
        "semantic_hash": config_hash(semantic),
        "input_data_hashes": {
            name: dataframe_content_hash(frame)
            for name, frame in sorted((dataframes or {}).items())
        },
        "source_hash": relevant_source_hash(source_paths, repo_root=root),
        "dependency_manifest_hash": dependency_manifest_hash(root),
    }


def output_fingerprints(
    base_dir: str | Path, relative_paths: Iterable[str | Path]
) -> dict[str, dict[str, Any]]:
    base = Path(base_dir)
    output: dict[str, dict[str, Any]] = {}
    for raw_path in relative_paths:
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Output path must stay below base directory: {relative}")
        record = file_fingerprint(base / relative)
        if record is None:
            raise FileNotFoundError(base / relative)
        output[relative.as_posix()] = record
    return output


def validate_artifact_manifest(
    manifest: Any,
    expected_fingerprint: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    required_outputs: Iterable[str | Path] = (),
) -> tuple[bool, str]:
    if not isinstance(manifest, Mapping):
        return False, "manifest is not an object"
    if manifest.get("fingerprint") != dict(expected_fingerprint):
        return False, "fingerprint mismatch"
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return False, "missing output fingerprints"
    required = {Path(path).as_posix() for path in required_outputs}
    if not required.issubset(outputs):
        return False, "missing required output fingerprint"
    if base_dir is None and outputs:
        return False, "output base directory unavailable"
    for relative_name, expected in outputs.items():
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            return False, f"unsafe output path: {relative_name}"
        actual = file_fingerprint(Path(base_dir) / relative)
        if actual is None:
            return False, f"missing output: {relative_name}"
        if actual != expected:
            return False, f"output fingerprint mismatch: {relative_name}"
    return True, "valid"


def forecast_trial_fingerprint(
    *,
    candidate: Mapping[str, Any],
    train_data: pd.DataFrame,
    model: str,
    epochs: int,
    seeds: Iterable[int],
    device: str,
    development_origins: Iterable[Any],
    benchmark_origins: Iterable[Any],
) -> dict[str, Any]:
    from anomaly_search_common import apply_candidate_config, selected_forecasting_config

    resolved_cfg = selected_forecasting_config()
    apply_candidate_config(resolved_cfg, dict(candidate))
    resolved_cfg.cv_epochs = int(epochs)
    resolved_cfg.final_epochs = int(epochs)
    resolved_cfg.seeds = tuple(int(seed) for seed in seeds)
    resolved_cfg.autoencoder_device = device
    semantic = {
        "candidate": candidate,
        "resolved_config": asdict(resolved_cfg),
        "model": model,
        "epochs": int(epochs),
        "seeds": [int(seed) for seed in seeds],
        "requested_device": device,
        "resolved_device": resolve_compute_device(device),
        "neural_training_identity": neural_training_identity(resolved_cfg),
        "development_origins": [pd.Timestamp(value).isoformat() for value in development_origins],
        "benchmark_origins": [pd.Timestamp(value).isoformat() for value in benchmark_origins],
    }
    return artifact_fingerprint(
        schema_version="anomaly-forecast-trial-v3",
        semantic=semantic,
        dataframes={"train": train_data},
        source_paths=FORECAST_TRIAL_SOURCE_PATHS,
    )


def diagnostic_trial_fingerprint(
    *,
    candidate: Mapping[str, Any],
    train_data: pd.DataFrame,
    cutoffs: Iterable[Any],
    seeds: Iterable[int],
    device: str,
    save_scores: bool,
) -> dict[str, Any]:
    semantic = {
        "candidate": candidate,
        "cutoffs": [pd.Timestamp(value).isoformat() for value in cutoffs],
        "seeds": [int(seed) for seed in seeds],
        "requested_device": device,
        "resolved_device": resolve_compute_device(device),
        "save_scores": bool(save_scores),
    }
    return artifact_fingerprint(
        schema_version="autoencoder-diagnostic-v3",
        semantic=semantic,
        dataframes={"train": train_data},
        source_paths=DIAGNOSTIC_TRIAL_SOURCE_PATHS,
    )


def _validate_result_body(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result_path: Path,
) -> tuple[bool, str]:
    schema = payload.get("schema_version")
    required_by_schema = {
        "anomaly-forecast-trial-v3": {
            "schema_version", "candidate", "model", "epochs", "seeds",
            "development_origins", "benchmark_origins", "development",
            "benchmark", "status",
        },
        "autoencoder-diagnostic-v3": {
            "schema_version", "candidate", "cutoffs", "seeds", "runs",
            "aggregate", "status",
        },
    }
    required = required_by_schema.get(schema)
    if required is None:
        return False, f"unsupported or unverifiable result schema: {schema!r}"
    missing = sorted(required - payload.keys())
    if missing:
        return False, f"result body is missing required fields: {missing}"
    expected_body = manifest.get("result_body")
    if not isinstance(expected_body, Mapping):
        return False, "missing canonical result-body digest"
    if dict(expected_body) != result_body_manifest(payload):
        return False, "canonical result-body digest mismatch"

    if schema != "anomaly-forecast-trial-v3":
        return True, "valid"
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return False, "missing output fingerprints"
    required_oofs = ("development_oof.parquet", "benchmark_oof.parquet")
    if not set(required_oofs).issubset(outputs):
        return False, "forecast result is missing authenticated OOF outputs"
    try:
        from anomaly_search_common import summarize_oof

        for split, origins_key in (
            ("development", "development_origins"),
            ("benchmark", "benchmark_origins"),
        ):
            frame = pd.read_parquet(result_path.parent / f"{split}_oof.parquet")
            expected_origins = {
                pd.Timestamp(value).date().isoformat() for value in payload[origins_key]
            }
            actual_origins = {
                pd.Timestamp(value).date().isoformat()
                for value in frame["origin"].dropna().unique()
            }
            if actual_origins != expected_origins:
                return False, f"{split} OOF origins do not match result body"
            recomputed = summarize_oof(frame, str(payload["model"]))
            if config_hash(recomputed) != config_hash(payload[split]):
                return False, f"{split} summary does not match authenticated OOF"
    except Exception as exc:
        return False, f"could not verify forecast OOF summaries: {exc}"
    return True, "valid"


def load_validated_result(
    result_path: str | Path,
    expected_fingerprint: Mapping[str, Any],
    *,
    required_outputs: Iterable[str | Path] = (),
) -> tuple[dict[str, Any] | None, str]:
    path = Path(result_path)
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable result: {exc}"
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        return None, "result is not complete"
    valid, reason = validate_artifact_manifest(
        payload.get("artifact_manifest"),
        expected_fingerprint,
        base_dir=path.parent,
        required_outputs=required_outputs,
    )
    if not valid:
        return None, reason
    body_valid, reason = _validate_result_body(
        payload, payload["artifact_manifest"], path
    )
    return (payload, "valid") if body_valid else (None, reason)
