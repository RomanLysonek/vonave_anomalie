from __future__ import annotations

import json

from anomaly_search_common import (
    CONFIG_FIELDS,
    PROFILES,
    autoencoder_action_variants,
    generate_autoencoder_candidates,
    generate_statistical_candidates,
)
from pipeline import configure_anomaly_runtime, parse_args
from framework import Config


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


def test_pipeline_can_load_winner_candidate_file(tmp_path) -> None:
    candidate = generate_autoencoder_candidates(1, seed=9, epoch_cap=3)[0]
    path = tmp_path / "winner.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    options = parse_args(["--anomaly-config", str(path)])
    cfg = Config()
    runtime = configure_anomaly_runtime(cfg, options)
    assert runtime["source"] == str(path)
    assert cfg.anomaly_source == "autoencoder"
    assert cfg.anomaly_mode == "features"
    assert cfg.autoencoder_max_epochs <= 3


def test_profiles_include_smoke_and_overnight() -> None:
    assert PROFILES["smoke"].autoencoder_epoch_cap < PROFILES["overnight"].autoencoder_epoch_cap
    assert PROFILES["overnight"].confirmation_seeds == (42, 123, 777)
