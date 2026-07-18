import pandas as pd
import pytest
import torch

from ml.framework import Config
from ml.pipeline import (
    _fold_checkpoint_signature,
    _guard_checkpoint_overwrite,
    _load_fold_checkpoint,
    _save_fold_checkpoint,
)


def test_fold_checkpoint_roundtrip_and_signature_guard(tmp_path):
    cfg = Config(num_products=2)
    origin = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame({"ProductId": [1], "prediction": [2.0]})
    timing = {"strategy": "recursive", "fold_seconds": 1.5}
    train = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": pd.to_datetime(["2023-12-30", "2023-12-31"]),
        "Quantity": [10.0, 11.0],
    })

    _save_fold_checkpoint(
        str(tmp_path), "recursive", "development", origin, cfg, train, frame, timing
    )
    loaded = _load_fold_checkpoint(
        str(tmp_path), "recursive", "development", origin, cfg, train
    )
    pd.testing.assert_frame_equal(loaded["oof"], frame)
    assert loaded["timing"] == timing

    incompatible = Config(num_products=3)
    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _load_fold_checkpoint(
            str(tmp_path), "recursive", "development", origin, incompatible, train
        )
    assert _load_fold_checkpoint(
        str(tmp_path), "recursive", "development", origin, incompatible, train,
        confirm_recompute_stale=True,
    ) is None

    changed = train.copy()
    changed.loc[0, "Quantity"] = 99.0
    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _load_fold_checkpoint(
            str(tmp_path), "recursive", "development", origin, cfg, changed
        )


def test_incomplete_c0_checkpoint_signature_is_rejected(tmp_path):
    import os
    import pickle
    from dataclasses import asdict
    from ml.pipeline import CHECKPOINT_SCHEMA_VERSION, _fold_checkpoint_path

    cfg = Config(num_products=2)
    origin = pd.Timestamp("2024-01-01")
    train = pd.DataFrame({
        "ProductId": [1],
        "DateKey": [pd.Timestamp("2023-12-31")],
        "Quantity": [1.0],
    })
    path = _fold_checkpoint_path(
        str(tmp_path), "direct", "development", origin
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    incomplete_cfg = asdict(cfg)
    incomplete_cfg.pop("reference_batch_size")
    payload = {
        "signature": {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "strategy": "direct",
            "origin_type": "development",
            "origin": origin.isoformat(),
            "cfg": incomplete_cfg,
        },
        "oof": pd.DataFrame({"ProductId": [1], "prediction": [2.0]}),
        "timing": {},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    with pytest.raises(RuntimeError, match="confirm-recompute-stale"):
        _load_fold_checkpoint(
            str(tmp_path), "direct", "development", origin, cfg, train
        )


def test_checkpoint_identity_tracks_backend_device_and_batch_semantics(monkeypatch):
    origin = pd.Timestamp("2024-01-01")
    train = pd.DataFrame({
        "ProductId": [1],
        "DateKey": [pd.Timestamp("2023-12-31")],
        "Quantity": [1.0],
    })
    base = Config(
        batch_size=256,
        reference_batch_size=512,
        nn_training_backend="auto",
        autoencoder_device="cpu",
    )
    signature = _fold_checkpoint_signature(
        base, "direct", "development", origin, train
    )
    assert signature != _fold_checkpoint_signature(
        Config(**{**base.__dict__, "nn_training_backend": "dataloader"}),
        "direct", "development", origin, train,
    )
    assert signature != _fold_checkpoint_signature(
        Config(**{**base.__dict__, "batch_size": 128}),
        "direct", "development", origin, train,
    )
    assert signature != _fold_checkpoint_signature(
        Config(**{**base.__dict__, "reference_batch_size": 1024}),
        "direct", "development", origin, train,
    )
    current_device = _fold_checkpoint_signature.__globals__["DEVICE"].type
    changed_device = "cpu" if current_device != "cpu" else "mps"
    monkeypatch.setitem(
        _fold_checkpoint_signature.__globals__, "DEVICE", torch.device(changed_device)
    )
    assert signature != _fold_checkpoint_signature(
        base, "direct", "development", origin, train
    )
    assert signature != _fold_checkpoint_signature(
        Config(**{**base.__dict__, "autoencoder_device": "mps"}),
        "direct", "development", origin, train,
    )


def test_checkpoint_overwrite_requires_explicit_confirmation(tmp_path) -> None:
    origin = pd.Timestamp("2024-01-01")
    path = tmp_path / "direct" / "development" / "2024-01-01.pkl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"expensive")

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        _guard_checkpoint_overwrite(
            str(tmp_path),
            "direct",
            "development",
            origin,
            resume=False,
            confirm_recompute_stale=False,
        )
    _guard_checkpoint_overwrite(
        str(tmp_path),
        "direct",
        "development",
        origin,
        resume=False,
        confirm_recompute_stale=True,
    )


def test_checkpoint_execution_identity_tampering_is_rejected(tmp_path) -> None:
    import pickle
    from ml.pipeline import _fold_checkpoint_path

    cfg = Config(num_products=1, seeds=(42,))
    origin = pd.Timestamp("2024-01-01")
    train = pd.DataFrame({
        "ProductId": [1],
        "DateKey": [pd.Timestamp("2023-12-31")],
        "Quantity": [1.0],
    })
    frame = pd.DataFrame({"ProductId": [1], "prediction": [2.0]})
    timing = {
        "neural_ran": False,
        "neural_training_stats": [],
    }
    _save_fold_checkpoint(
        str(tmp_path), "direct", "development", origin, cfg, train, frame, timing
    )
    path = _fold_checkpoint_path(str(tmp_path), "direct", "development", origin)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    payload["timing"]["neural_ran"] = True
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(RuntimeError, match="execution identity"):
        _load_fold_checkpoint(
            str(tmp_path), "direct", "development", origin, cfg, train
        )
