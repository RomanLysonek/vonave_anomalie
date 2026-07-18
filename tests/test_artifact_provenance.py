from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from artifact_provenance import (
    artifact_fingerprint,
    canonical_json,
    dataframe_content_hash,
    file_fingerprint,
    file_hash,
    FORECAST_TRIAL_SOURCE_PATHS,
    load_validated_result,
    output_fingerprints,
    relevant_source_hash,
    result_body_manifest,
    validate_artifact_manifest,
)
from anomaly_search_common import summarize_oof, validate_target_roles


def test_canonical_and_dataframe_hashes_are_content_sensitive() -> None:
    assert canonical_json({"b": 2, "a": [1, pd.Timestamp("2026-01-01")]}) == (
        '{"a":[1,"2026-01-01T00:00:00"],"b":2}'
    )
    left = pd.DataFrame({
        "ProductId": [1, 2],
        "DateKey": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Quantity": [3.0, 4.0],
    })
    right = left.copy()
    right.loc[0, "Quantity"] = 30.0
    assert dataframe_content_hash(left) != dataframe_content_hash(right)


def test_fingerprint_tracks_source_config_and_output_integrity(tmp_path) -> None:
    source = tmp_path / "model.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    data = pd.DataFrame({"x": [1, 2]})
    first = artifact_fingerprint(
        schema_version="test-v1",
        semantic={"candidate": {"alpha": 1}},
        dataframes={"train": data},
        source_paths=(source,),
    )
    changed_config = artifact_fingerprint(
        schema_version="test-v1",
        semantic={"candidate": {"alpha": 2}},
        dataframes={"train": data},
        source_paths=(source,),
    )
    assert first != changed_config

    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed_source = artifact_fingerprint(
        schema_version="test-v1",
        semantic={"candidate": {"alpha": 1}},
        dataframes={"train": data},
        source_paths=(source,),
    )
    assert first != changed_source

    output = tmp_path / "artifact.bin"
    output.write_bytes(b"valid")
    manifest = {
        "fingerprint": changed_source,
        "outputs": output_fingerprints(tmp_path, ("artifact.bin",)),
    }
    assert validate_artifact_manifest(
        manifest,
        changed_source,
        base_dir=tmp_path,
        required_outputs=("artifact.bin",),
    ) == (True, "valid")
    output.write_bytes(b"corrupt")
    valid, reason = validate_artifact_manifest(
        json.loads(json.dumps(manifest)),
        changed_source,
        base_dir=tmp_path,
        required_outputs=("artifact.bin",),
    )
    assert not valid
    assert "output fingerprint mismatch" in reason


def test_forecast_result_body_is_bound_and_recomputed_from_authenticated_oof(
    tmp_path,
) -> None:
    frames = {}
    for split, origin in (
        ("development", pd.Timestamp("2025-01-01")),
        ("benchmark", pd.Timestamp("2026-01-01")),
    ):
        frames[split] = pd.DataFrame({
            "ProductId": [1, 2],
            "origin": [origin, origin],
            "horizon": [1, 1],
            "actual": [10.0, 20.0],
            "pred_DynamicRidge": [9.0, 18.0],
            "ProductAvailable": [True, True],
        })
        frames[split].to_parquet(tmp_path / f"{split}_oof.parquet", index=False)
    fingerprint = artifact_fingerprint(
        schema_version="forecast-test-v1",
        semantic={"candidate": "one"},
        source_paths=(),
    )
    payload = {
        "schema_version": "anomaly-forecast-trial-v4",
        "candidate": {"id": "one", "family": "control"},
        "model": "DynamicRidge",
        "epochs": 1,
        "seeds": [42],
        "development_origins": ["2025-01-01"],
        "benchmark_origins": ["2026-01-01"],
        "target_role_validation": validate_target_roles(
            development_origins=["2025-01-01"],
            benchmark_origins=["2026-01-01"],
        ),
        "development": summarize_oof(frames["development"], "DynamicRidge"),
        "benchmark": summarize_oof(frames["benchmark"], "DynamicRidge"),
        "status": "complete",
    }
    payload["artifact_manifest"] = {
        "fingerprint": fingerprint,
        "outputs": output_fingerprints(
            tmp_path, ("development_oof.parquet", "benchmark_oof.parquet")
        ),
        "result_body": result_body_manifest(payload),
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_validated_result(result_path, fingerprint)[0] is not None

    payload["development"]["global"]["WAPE"] = 999.0
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    _, reason = load_validated_result(result_path, fingerprint)
    assert "result-body digest mismatch" in reason

    payload["artifact_manifest"]["result_body"] = result_body_manifest(payload)
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    _, reason = load_validated_result(result_path, fingerprint)
    assert "summary does not match authenticated OOF" in reason


def test_explicit_source_set_ignores_publication_only_changes(tmp_path) -> None:
    assert all(path.name != "publish_site.py" for path in FORECAST_TRIAL_SOURCE_PATHS)
    source = tmp_path / "training.py"
    publication = tmp_path / "publish_site.py"
    source.write_text("TRAINING = 1\n", encoding="utf-8")
    publication.write_text("PUBLISH = 1\n", encoding="utf-8")
    before = relevant_source_hash((source,), repo_root=tmp_path)
    publication.write_text("PUBLISH = 2\n", encoding="utf-8")
    assert relevant_source_hash((source,), repo_root=tmp_path) == before


def test_diagnostic_result_without_canonical_body_digest_is_rejected(tmp_path) -> None:
    fingerprint = artifact_fingerprint(
        schema_version="diagnostic-test-v1",
        semantic={"candidate": "one"},
        source_paths=(),
    )
    payload = {
        "schema_version": "autoencoder-diagnostic-v4",
        "candidate": {"id": "one"},
        "cutoffs": ["2026-01-01"],
        "seeds": [42],
        "runs": [{"seed": 42}],
        "aggregate": {"diagnostic_objective": 1.0},
        "diagnostic_boundary": {"schema_version": "development-diagnostic-boundary-v2"},
        "status": "complete",
        "artifact_manifest": {"fingerprint": fingerprint, "outputs": {}},
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, reason = load_validated_result(path, fingerprint)
    assert reason == "missing canonical result-body digest"


def test_artifact_fingerprints_reject_symlinks_and_nonregular_files(tmp_path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"content")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    assert file_hash(link) is None
    assert file_fingerprint(link) is None

    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is not supported")
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    assert file_hash(fifo) is None
    assert file_fingerprint(fifo) is None
