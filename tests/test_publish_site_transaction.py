from __future__ import annotations

import os
from pathlib import Path

import pytest

import ml.publish_site as publication


DESTINATIONS = (
    Path("outputs/results.json"),
    Path("outputs/dashboard"),
    Path("docs"),
    Path("outputs/published_results_manifest.json"),
    Path("outputs/dashboard_manifest.json"),
)


def _write_entry(root: Path, relative: Path, value: str) -> None:
    path = root / relative
    if relative.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)
        (path / "payload.txt").write_text(value, encoding="utf-8")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("failed_phase", range(len(DESTINATIONS)))
def test_every_commit_phase_rolls_back_all_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_phase: int
) -> None:
    root = tmp_path / "repository"
    stage = tmp_path / "stage"
    for relative in DESTINATIONS:
        _write_entry(root, relative, f"old-{relative}")
        _write_entry(stage, relative, f"new-{relative}")
    before = _snapshot(root)
    calls = 0
    real_install = publication._install_staged_destination

    def fail_selected(source: Path, destination: Path) -> None:
        nonlocal calls
        phase = calls
        calls += 1
        if phase == failed_phase:
            raise OSError(f"injected commit failure {phase}")
        real_install(source, destination)

    monkeypatch.setattr(publication, "_install_staged_destination", fail_selected)
    with pytest.raises(OSError, match="injected commit failure"):
        publication._commit_publication(stage, root)

    assert _snapshot(root) == before
    assert not any(path.name == ".publication-backups" for path in tmp_path.rglob("*"))


def test_custom_results_and_canonical_results_survive_staging_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    custom = root / "custom-results.json"
    canonical = root / "outputs" / "results.json"
    custom.parent.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    custom.write_text('{"custom": true}\n', encoding="utf-8")
    canonical.write_text('{"canonical": true}\n', encoding="utf-8")

    def fail_validation(*args, **kwargs):
        raise RuntimeError("injected staging validation failure")

    monkeypatch.setattr(publication, "_build_publication_stage", fail_validation)
    with pytest.raises(RuntimeError, match="staging validation"):
        publication.publish_site(root, custom)

    assert custom.read_text(encoding="utf-8") == '{"custom": true}\n'
    assert canonical.read_text(encoding="utf-8") == '{"canonical": true}\n'


@pytest.mark.parametrize(
    "relative",
    [
        "outputs/published_results_manifest.json",
        "outputs/dashboard_manifest.json",
        "docs/data/results.json",
        "outputs/dashboard/results.json",
    ],
)
def test_custom_results_cannot_overlap_publication_destinations(
    tmp_path: Path, relative: str
) -> None:
    root = tmp_path / "repository"
    custom = root / relative
    custom.parent.mkdir(parents=True, exist_ok=True)
    custom.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must not overlap"):
        publication.publish_site(root, custom)


def test_post_commit_validation_failure_restores_every_backup(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    stage = tmp_path / "stage"
    for relative in DESTINATIONS:
        _write_entry(root, relative, f"old-{relative}")
        _write_entry(stage, relative, f"new-{relative}")
    before = _snapshot(root)

    with pytest.raises(RuntimeError, match="post-commit"):
        publication._commit_publication(
            stage,
            root,
            lambda: (_ for _ in ()).throw(RuntimeError("post-commit validation failed")),
        )

    assert _snapshot(root) == before


@pytest.mark.parametrize("kind", ["file-symlink", "directory-symlink", "broken-symlink"])
def test_authored_static_rejects_all_symlinks(tmp_path: Path, kind: str) -> None:
    static = tmp_path / "static"
    static.mkdir()
    if kind == "file-symlink":
        target = tmp_path / "target.js"
        target.write_text("safe", encoding="utf-8")
        (static / "app.js").symlink_to(target)
    elif kind == "directory-symlink":
        target = tmp_path / "target-dir"
        target.mkdir()
        (static / "linked-dir").symlink_to(target, target_is_directory=True)
    else:
        (static / "broken.js").symlink_to(tmp_path / "missing.js")

    with pytest.raises(RuntimeError, match="symlink"):
        publication._regular_tree_files(static, "Authored static")


def test_existing_docs_rejects_fifo_when_supported(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is not supported")
    docs = tmp_path / "docs"
    docs.mkdir()
    os.mkfifo(docs / "pipe")

    with pytest.raises(RuntimeError, match="non-regular"):
        publication._assert_replace_safe(docs)
