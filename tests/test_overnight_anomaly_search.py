from __future__ import annotations

import json
from argparse import Namespace

import pandas as pd
import pytest

from anomaly_search_common import (
    CONFIG_FIELDS,
    PROFILES,
    apply_candidate_config,
    autoencoder_action_variants,
    generate_autoencoder_candidates,
    generate_statistical_candidates,
    development_diagnostic_boundary,
    target_dates_for_origins,
    validate_target_roles,
    validate_development_diagnostic_boundary,
)
from pipeline import configure_anomaly_runtime, parse_args
from framework import Config
from artifact_provenance import (
    artifact_fingerprint,
    output_fingerprints,
    result_body_manifest,
)
from run_overnight_anomaly_search import (
    _diagnostic_cutoffs,
    _confirmation_recommendation,
    _invalidate_legacy_execution_artifacts,
    _profile_target_roles as overnight_profile_target_roles,
    _rank_forecast_results,
    _should_skip,
)
from run_weekend_v2_search import _profile_target_roles as weekend_profile_target_roles
from weekend_v2_common import WEEKEND_V2_PROFILES


def test_candidate_generation_is_deterministic_and_config_valid() -> None:
    left = generate_autoencoder_candidates(6, seed=123, epoch_cap=12)
    right = generate_autoencoder_candidates(6, seed=123, epoch_cap=12)
    assert left == right
    assert len({item["id"] for item in left}) == 6
    for item in left:
        assert set(item["config"]) <= CONFIG_FIELDS
        assert item["config"]["autoencoder_max_epochs"] <= 12

    statistical = generate_statistical_candidates(5, seed=456)
    assert len({item["id"] for item in statistical}) == 5
    assert all(set(item["config"]) <= CONFIG_FIELDS for item in statistical)
    with pytest.raises(ValueError, match="unknown Config field"):
        apply_candidate_config(
            Config(),
            {"config": {"allow_autoencoder_cache_build": True}},
        )


def test_autoencoder_action_variants_preserve_model_and_change_action() -> None:
    base = generate_autoencoder_candidates(1, seed=1, epoch_cap=4)[0]
    variants = autoencoder_action_variants(base)
    assert {item["config"]["anomaly_mode"] for item in variants} == {"features", "both"}
    stripped = []
    for item in variants:
        config = dict(item["config"])
        config.pop("anomaly_mode")
        stripped.append(config)
    assert stripped[0] == stripped[1]


def test_pipeline_rejects_archived_overnight_recommendation(tmp_path) -> None:
    path = tmp_path / "recommendation.json"
    path.write_text(json.dumps({
        "schema_version": "overnight-anomaly-search-v4",
        "status": "archived",
        "provenance_status": "unverified",
        "execution_enabled": False,
        "winner": generate_autoencoder_candidates(1, seed=9, epoch_cap=3)[0],
    }), encoding="utf-8")
    options = parse_args(["--anomaly-config", str(path)])
    with pytest.raises(ValueError, match="manual-anomaly-config-v1"):
        configure_anomaly_runtime(Config(), options)


def test_pipeline_accepts_only_explicit_manual_anomaly_config(tmp_path) -> None:
    path = tmp_path / "manual.json"
    path.write_text(json.dumps({
        "schema_version": "manual-anomaly-config-v1",
        "config": {"anomaly_mode": "off"},
    }), encoding="utf-8")
    options = parse_args(["--anomaly-config", str(path)])
    cfg = Config()
    configure_anomaly_runtime(cfg, options)
    assert cfg.anomaly_mode == "off"


def test_orchestrator_invalidates_legacy_execution_files_at_startup(tmp_path) -> None:
    for name in ("recommendation.json", "winner_candidate.json"):
        (tmp_path / name).write_text('{"execution_enabled": true}', encoding="utf-8")
    _invalidate_legacy_execution_artifacts(tmp_path)
    assert not (tmp_path / "recommendation.json").exists()
    assert not (tmp_path / "winner_candidate.json").exists()
    marker = json.loads((tmp_path / "execution_disabled.json").read_text())
    assert marker["execution_enabled"] is False


def test_profiles_include_smoke_and_overnight() -> None:
    assert PROFILES["smoke"].autoencoder_epoch_cap < PROFILES["overnight"].autoencoder_epoch_cap
    assert PROFILES["overnight"].confirmation_seeds == (42, 123, 777)


def test_orchestrator_only_skips_valid_fingerprinted_result(tmp_path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    output = trial / "scores.parquet"
    output.write_bytes(b"scores")
    data = pd.DataFrame({"Quantity": [1.0, 2.0]})
    expected = artifact_fingerprint(
        schema_version="trial-v2",
        semantic={"candidate": {"id": "one"}},
        dataframes={"train": data},
        source_paths=(),
    )
    payload = {
        "schema_version": "autoencoder-diagnostic-v4",
        "candidate": {"id": "one"},
        "cutoffs": ["2026-01-01"],
        "seeds": [42],
        "runs": [{"seed": 42}],
        "aggregate": {"diagnostic_objective": 1.0},
        "status": "complete",
        "diagnostic_boundary": {"schema_version": "test"},
    }
    payload["artifact_manifest"] = {
        "fingerprint": expected,
        "outputs": output_fingerprints(trial, ("scores.parquet",)),
        "result_body": result_body_manifest(payload),
    }
    (trial / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    args = Namespace(retry_failed=False, confirm_recompute_stale=False)
    assert _should_skip(
        trial,
        args,
        expected,
        required_outputs=("scores.parquet",),
    )

    stale = artifact_fingerprint(
        schema_version="trial-v2",
        semantic={"candidate": {"id": "two"}},
        dataframes={"train": data},
        source_paths=(),
    )
    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _should_skip(trial, args, stale, required_outputs=("scores.parquet",))

    output.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _should_skip(trial, args, expected, required_outputs=("scores.parquet",))
    confirmed = Namespace(retry_failed=False, confirm_recompute_stale=True)
    assert not _should_skip(
        trial, confirmed, expected, required_outputs=("scores.parquet",)
    )

    (trial / "result.json").unlink()
    (trial / "failure.json").write_text(
        json.dumps({"status": "failed", "fingerprint": expected}),
        encoding="utf-8",
    )
    assert _should_skip(trial, args, expected)
    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _should_skip(trial, args, stale)

    (trial / "result.json").write_text(
        json.dumps({"status": "complete", "artifact_manifest": {}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _should_skip(trial, args, expected)


def test_forecast_ranking_is_invariant_to_frozen_benchmark() -> None:
    def payload(candidate_id: str, family: str, dev: float, bench: float) -> dict:
        summary = lambda wape: {
            "global": {"WAPE": wape, "BiasRatio": 0.0},
            "top_actual_decile": {"WAPE": wape},
        }
        return {
            "candidate": {"id": candidate_id, "name": candidate_id, "family": family},
            "model": "NeuralNet",
            "development": summary(dev),
            "benchmark": summary(bench),
        }

    results = [
        payload("control", "control", 1.0, 1.0),
        payload("a", "statistical", 0.8, 100.0),
        payload("b", "statistical", 0.9, 0.01),
    ]
    assert [row["candidate_id"] for row in _rank_forecast_results(results)] == [
        "a", "b", "control"
    ]
    results[1]["benchmark"] = {
        "global": {"WAPE": 0.0001, "BiasRatio": 0.0},
        "top_actual_decile": {"WAPE": 0.0001},
    }
    results[2]["benchmark"] = {
        "global": {"WAPE": 9999.0, "BiasRatio": 0.0},
        "top_actual_decile": {"WAPE": 9999.0},
    }
    assert [row["candidate_id"] for row in _rank_forecast_results(results)] == [
        "a", "b", "control"
    ]


def test_diagnostic_boundary_precedes_frozen_benchmark_and_rejects_overlap() -> None:
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    train = pd.DataFrame({
        "DateKey": dates,
        "ProductId": 1,
        "Quantity": 1.0,
    })
    boundary = development_diagnostic_boundary(
        train, pd.to_datetime(["2025-03-20", "2025-03-27"])
    )
    cutoffs = _diagnostic_cutoffs(train, 3, boundary)
    source = validate_development_diagnostic_boundary(train, boundary, cutoffs)
    assert pd.Timestamp(source["DateKey"].max()) == pd.Timestamp("2025-03-19")
    assert cutoffs.max() + pd.Timedelta(days=7) < pd.Timestamp("2025-03-20")
    assert boundary["source_partition"] == "train_data_development_only"
    assert len(boundary["source_content_sha256"]) == 64

    with pytest.raises(ValueError, match="overlaps"):
        validate_development_diagnostic_boundary(
            train, boundary, [pd.Timestamp("2025-03-13")]
        )


def test_target_role_boundaries_and_checked_data_february_overlap() -> None:
    adjacent = validate_target_roles(
        development_origins=["2025-02-01"],
        benchmark_origins=["2025-02-08"],
        frozen_final_origins=[],
        horizon=7,
    )
    assert adjacent["roles"]["development"]["target_end"] == "2025-02-08"
    assert adjacent["roles"]["benchmark"]["target_start"] == "2025-02-09"
    assert len(adjacent["content_sha256"]) == 64
    assert target_dates_for_origins(["2025-02-01"], 7)[-1] == pd.Timestamp(
        "2025-02-08"
    )

    with pytest.raises(ValueError, match="development and benchmark"):
        validate_target_roles(
            development_origins=["2025-02-10"],
            benchmark_origins=["2025-02-09"],
            frozen_final_origins=[],
            horizon=7,
        )


@pytest.mark.parametrize("horizon", [1, 2, 7, 14])
def test_target_role_boundary_property_adjacent_ok_one_day_overlap_fails(
    horizon: int,
) -> None:
    development = pd.Timestamp("2024-03-01")
    adjacent_benchmark = development + pd.Timedelta(days=horizon)
    validate_target_roles(
        development_origins=[development],
        benchmark_origins=[adjacent_benchmark],
        frozen_final_origins=[],
        horizon=horizon,
    )
    with pytest.raises(ValueError, match="Target-role overlap"):
        validate_target_roles(
            development_origins=[development],
            benchmark_origins=[adjacent_benchmark - pd.Timedelta(days=1)],
            frozen_final_origins=[],
            horizon=horizon,
        )
    with pytest.raises(ValueError, match="development and calibration"):
        validate_target_roles(
            development_origins=[development],
            calibration_origins=[development],
            benchmark_origins=[],
            frozen_final_origins=[],
            horizon=horizon,
        )


def test_every_search_profile_is_prevalidated_and_overlap_is_rejected() -> None:
    train = pd.DataFrame({
        "DateKey": pd.date_range("2021-01-01", "2026-01-11", freq="D"),
        "ProductId": 1,
        "Quantity": 1.0,
    })
    for name, profile in PROFILES.items():
        maximum = max(
            profile.proxy_benchmark_origins,
            profile.neural_benchmark_origins,
            profile.confirmation_benchmark_origins,
        )
        boundary = development_diagnostic_boundary(
            train,
            pd.DatetimeIndex([
                train["DateKey"].max() - pd.Timedelta(days=7 * (index + 1))
                for index in range(maximum)
            ]),
        )
        if name == "exhaustive":
            with pytest.raises(ValueError, match="Target-role overlap"):
                overnight_profile_target_roles(train, profile, boundary)
        else:
            validated = overnight_profile_target_roles(train, profile, boundary)
            assert set(validated["stages"]) == {
                "diagnostic", "proxy", "neural", "confirmation"
            }
    for name, profile in WEEKEND_V2_PROFILES.items():
        if name == "exhaustive-v2":
            with pytest.raises(ValueError, match="Target-role overlap"):
                weekend_profile_target_roles(train, profile)
        else:
            validated = weekend_profile_target_roles(train, profile)
            assert set(validated["stages"]) == {
                "screen", "refine", "confirmation"
            }


def test_confirmation_acceptance_and_winner_ignore_benchmark(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(
        _confirmation_recommendation.__globals__,
        "bootstrap_origin_improvement",
        lambda *args, **kwargs: {"probability_improvement_positive": 1.0},
    )

    def result(candidate_id: str, family: str, dev: float, bench: float) -> dict:
        trial = tmp_path / candidate_id
        trial.mkdir()
        pd.DataFrame({
            "ProductId": [1],
            "DateKey": [pd.Timestamp("2026-01-02")],
            "origin": [pd.Timestamp("2026-01-01")],
            "actual": [10.0],
            "pred_NeuralNet": [9.0],
            "ProductAvailable": [True],
        }).to_parquet(trial / "development_oof.parquet", index=False)
        summary = lambda wape: {
            "global": {"WAPE": wape},
            "top_actual_decile": {"WAPE": wape},
            "by_stratum": {},
        }
        return {
            "_result_path": str(trial / "result.json"),
            "candidate": {
                "id": candidate_id,
                "name": candidate_id,
                "family": family,
            },
            "development": summary(dev),
            "benchmark": summary(bench),
        }

    results = [
        result("control", "control", 1.0, 1.0),
        result("candidate", "statistical", 0.8, 1000.0),
    ]
    before = _confirmation_recommendation(tmp_path, results)
    results[1]["benchmark"]["global"]["WAPE"] = 0.00001
    after = _confirmation_recommendation(tmp_path, results)
    assert before["winner"] == after["winner"]
    assert before["comparisons"][0]["accepted"] == after["comparisons"][0]["accepted"]
    assert before["status"] == "archived"
    assert before["provenance_status"] == "unverified"
    assert before["execution_enabled"] is False
    assert "final_submission_command" not in before
    assert not (tmp_path / "winner_candidate.json").exists()
