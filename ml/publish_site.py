"""Build and safely publish the checked-in static dashboard.

Run ``uv run python ml/publish_site.py`` to regenerate ``docs/`` and published
artifacts. Run with ``--check`` to build in repository-local staging directories
and verify byte-for-byte parity without replacing the site.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.artifact_provenance import (  # noqa: E402
    config_hash,
    dependency_manifest_hash,
    file_fingerprint,
    file_hash,
    output_fingerprints,
    relevant_source_hash,
)
from scripts.verify_published_site import verify_publication  # noqa: E402
from webapp.anomaly_dashboard import publish_anomaly_artifacts  # noqa: E402


SITE_SCHEMA_VERSION = "static-dashboard-site-v2"
DASHBOARD_MANIFEST_SCHEMA = "dashboard-publication-v2"
PUBLISHED_RESULTS_SCHEMA = "published-results-v4"
LEGACY_GENERATED_FILES = {
    ".nojekyll",
    "README.md",
    "app.js",
    "common.js",
    "dataset.html",
    "dataset.js",
    "evaluation.html",
    "evaluation.js",
    "favicon.svg",
    "index.html",
    "model.html",
    "model.js",
    "results.json",
    "styles.css",
}
AUTHORED_SITE_FILES = {
    "anomalies.js",
    "app.js",
    "common.js",
    "dataset.html",
    "dataset.js",
    "evaluation.html",
    "evaluation.js",
    "favicon.svg",
    "index.html",
    "model.html",
    "model.js",
    "styles.css",
}


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _deterministic_environment(root: Path) -> dict[str, Any]:
    return {
        "recorded": False,
        "git": {"commit": None, "dirty": None},
        "python": None,
        "platform": None,
        "packages": {},
        "reason": (
            "Volatile build-host and Git state are intentionally unknown in the "
            "deterministic checked-in publication; content hashes are canonical."
        ),
        "dependency_manifest_hash": dependency_manifest_hash(root),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _regular_tree_files(root: Path, label: str) -> list[Path]:
    try:
        info = root.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} tree is missing: {root}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"{label} tree is not a real directory: {root}")
    files: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                entry_info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_info.st_mode):
                    raise RuntimeError(f"{label} tree contains a symlink: {path}")
                if stat.S_ISDIR(entry_info.st_mode):
                    visit(path)
                elif stat.S_ISREG(entry_info.st_mode):
                    files.append(path)
                else:
                    raise RuntimeError(f"{label} tree contains a non-regular entry: {path}")

    visit(root)
    return sorted(files)


def _regular_file(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(path) from exc
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")


def _site_files(root: Path, *, include_manifest: bool = False) -> list[Path]:
    files = _regular_tree_files(root, "Generated docs")
    if not include_manifest:
        files = [path for path in files if path.relative_to(root).as_posix() != "site-manifest.json"]
    return sorted(files)


def _hash_tree(root: Path, *, include_manifest: bool = True) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): str(file_hash(path))
        for path in _site_files(root, include_manifest=include_manifest)
    }


def _owned_paths(docs_dir: Path) -> set[str]:
    manifest_path = docs_dir / "site-manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest.get("files", [])
            owned = {
                str(record["path"])
                for record in records
                if isinstance(record, dict) and isinstance(record.get("path"), str)
            }
        except (OSError, json.JSONDecodeError, TypeError):
            raise RuntimeError("Existing docs/site-manifest.json is unreadable; refusing replacement")
        owned.add("site-manifest.json")
        return owned
    actual = {
        path.relative_to(docs_dir).as_posix()
        for path in _regular_tree_files(docs_dir, "Existing docs")
    }
    if actual.issubset(LEGACY_GENERATED_FILES):
        return actual
    raise RuntimeError(
        "Existing docs/ has no generated ownership manifest; refusing replacement"
    )


def _assert_replace_safe(docs_dir: Path) -> None:
    if not _path_exists(docs_dir):
        return
    files = _regular_tree_files(docs_dir, "Existing docs")
    actual = {
        path.relative_to(docs_dir).as_posix()
        for path in files
    }
    unowned = actual - _owned_paths(docs_dir)
    if unowned:
        raise RuntimeError(
            "Refusing to replace docs/ because it contains unowned files: "
            + ", ".join(sorted(unowned))
        )


def _remove_known_tree(path: Path) -> None:
    if not _path_exists(path):
        return
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        path.unlink()
        return
    with os.scandir(path) as entries:
        children = [Path(entry.path) for entry in entries]
    for child in children:
        _remove_known_tree(child)
    path.rmdir()


def _static_html(source: str) -> str:
    result = source.replace('href="/static/', 'href="./').replace('src="/static/', 'src="./')
    result = result.replace('href="/dataset"', 'href="./dataset.html"')
    result = result.replace('href="/evaluation"', 'href="./evaluation.html"')
    marker = '<script src="./common.js'
    if marker in result and "window.STATIC_DASHBOARD" not in result:
        result = result.replace(
            marker,
            '<script>window.STATIC_DASHBOARD = true;</script>\n  ' + marker,
            1,
        )
    return result


def _copy_authored_site(static_dir: Path, stage: Path) -> None:
    sources = _regular_tree_files(static_dir, "Authored static")
    if any(source.parent != static_dir for source in sources):
        raise RuntimeError("Authored static tree must contain regular files at its root only")
    actual = {source.name for source in sources}
    unexpected = actual - AUTHORED_SITE_FILES
    if unexpected:
        raise RuntimeError(
            "Refusing to publish unexpected authored-site files: "
            + ", ".join(sorted(unexpected))
        )
    for name in sorted(AUTHORED_SITE_FILES):
        source = static_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = stage / source.name
        if source.suffix.lower() == ".html":
            destination.write_text(_static_html(source.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copyfile(source, destination)
    (stage / ".nojekyll").write_text("", encoding="utf-8")


def _published_manifest(
    root: Path,
    anomaly_dir: Path,
    result_path: Path,
    aggregate: dict[str, Any],
    site_hashes: dict[str, str],
    captured_environment: dict[str, Any],
) -> dict[str, Any]:
    anomaly_paths = [
        path.relative_to(anomaly_dir)
        for path in sorted(anomaly_dir.rglob("*.json"))
    ]
    return {
        "schema_version": PUBLISHED_RESULTS_SCHEMA,
        "generated_at": aggregate["generated_at"],
        "generated_at_basis": aggregate["generated_at_basis"],
        "snapshot_as_of": aggregate["snapshot_as_of"],
        "artifact_schema_version": aggregate["schema_version"],
        "source_manifest_hash": aggregate["source_manifest_hash"],
        "source_hash": relevant_source_hash(
            (
                root / "webapp" / "anomaly_dashboard.py",
                root / "ml" / "publish_site.py",
                root / "ml" / "anomaly_detection.py",
                root / "ml" / "artifact_provenance.py",
            ),
            repo_root=root,
        ),
        "data_hashes": {
            record["path"]: record["sha256"]
            for record in aggregate.get("source_inputs", [])
        },
        "input_data_hash": config_hash({
            record["path"]: record["sha256"]
            for record in aggregate.get("source_inputs", [])
        }),
        "configuration_hash": config_hash({
            "recommendation": aggregate["recommendation"],
            "research_status": aggregate["research_status"],
        }),
        "dependency_manifest_hash": dependency_manifest_hash(root),
        "forecast_results": {
            "path": "outputs/results.json",
            **(file_fingerprint(result_path) or {"sha256": None, "size": None}),
        },
        "outputs": output_fingerprints(anomaly_dir, anomaly_paths),
        "site_artifacts": site_hashes,
        "excluded_evidence": aggregate["excluded_evidence"],
        "environment": captured_environment,
        "provenance_limitations": [
            "No anomaly truth labels exist.",
            "Historical overnight, diagnostic, and search artifacts may contain benchmark-target contamination; scientific status is contaminated and provenance is unverified.",
            "Legacy overnight/search/preflight artifacts are excluded from all current candidate and selection evidence.",
        ],
    }


def _stage_site(
    root: Path,
    stage: Path,
    anomaly_dir: Path,
    aggregate: dict[str, Any],
    *,
    results_path: Path,
    captured_environment: dict[str, Any],
) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=True)
    _copy_authored_site(root / "webapp" / "static", stage)
    data_dir = stage / "data"
    data_dir.mkdir()
    shutil.copyfile(results_path, data_dir / "results.json")
    shutil.copyfile(
        anomaly_dir / "anomaly-dashboard-v2.json",
        data_dir / "anomaly-dashboard-v2.json",
    )
    shutil.copytree(
        anomaly_dir / "anomaly-products-v2",
        data_dir / "anomaly-products-v2",
    )

    preliminary_hashes = _hash_tree(stage, include_manifest=False)
    published_manifest = _published_manifest(
        root,
        anomaly_dir,
        results_path,
        aggregate,
        preliminary_hashes,
        captured_environment,
    )
    _write_json(data_dir / "published_results_manifest.json", published_manifest)

    records = [
        {
            "path": path.relative_to(stage).as_posix(),
            **(file_fingerprint(path) or {}),
        }
        for path in _site_files(stage)
    ]
    site_manifest = {
        "schema_version": SITE_SCHEMA_VERSION,
        "generated_at": aggregate["generated_at"],
        "generated_at_basis": aggregate["generated_at_basis"],
        "source_manifest_hash": aggregate["source_manifest_hash"],
        "files": records,
    }
    _write_json(stage / "site-manifest.json", site_manifest)
    return published_manifest


def _compare_trees(expected: Path, actual: Path, label: str) -> None:
    expected_hashes = _hash_tree(expected)
    actual_hashes = _hash_tree(actual)
    if expected_hashes != actual_hashes:
        missing = sorted(set(expected_hashes) - set(actual_hashes))
        extra = sorted(set(actual_hashes) - set(expected_hashes))
        changed = sorted(
            path
            for path in set(expected_hashes) & set(actual_hashes)
            if expected_hashes[path] != actual_hashes[path]
        )
        raise RuntimeError(
            f"{label} parity failed; missing={missing}, extra={extra}, changed={changed}"
        )


def _compare_files(expected: Path, actual: Path, label: str) -> None:
    if file_fingerprint(expected) != file_fingerprint(actual):
        raise RuntimeError(f"{label} parity failed")


def _dashboard_manifest(
    publication_root: Path,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    dashboard = publication_root / "outputs" / "dashboard"
    product_files = sorted(
        (dashboard / "anomaly-products-v2").glob("product-*.json")
    )
    product_fingerprints = {
        path.relative_to(publication_root).as_posix(): file_fingerprint(path)
        for path in product_files
    }
    return {
        "schema_version": DASHBOARD_MANIFEST_SCHEMA,
        "entrypoint": "docs/index.html",
        "static_site": "docs",
        "forecast_results": {
            "path": "outputs/results.json",
            **(file_fingerprint(publication_root / "outputs" / "results.json") or {}),
        },
        "anomaly_aggregate": {
            "path": "outputs/dashboard/anomaly-dashboard-v2.json",
            **(file_fingerprint(dashboard / "anomaly-dashboard-v2.json") or {}),
        },
        "anomaly_products": {
            "path_template": "outputs/dashboard/anomaly-products-v2/product-{product_id}.json",
            "count": len(product_files),
            "sha256": config_hash(product_fingerprints),
            "files": product_fingerprints,
        },
        "product_count": len(product_files),
        "source_manifest_hash": aggregate["source_manifest_hash"],
        "site_manifest": {
            "path": "docs/site-manifest.json",
            **(file_fingerprint(publication_root / "docs" / "site-manifest.json") or {}),
        },
        "published_results_manifest": {
            "path": "outputs/published_results_manifest.json",
            **(
                file_fingerprint(
                    publication_root / "outputs" / "published_results_manifest.json"
                )
                or {}
            ),
        },
    }


def _build_publication_stage(
    root: Path,
    stage: Path,
    results: Path,
    captured_environment: dict[str, Any],
) -> dict[str, Any]:
    staged_outputs = stage / "outputs"
    staged_outputs.mkdir(parents=True)
    staged_results = staged_outputs / "results.json"
    shutil.copyfile(results, staged_results)
    anomaly_stage = staged_outputs / "dashboard"
    aggregate = publish_anomaly_artifacts(root, anomaly_stage)
    published = _stage_site(
        root,
        stage / "docs",
        anomaly_stage,
        aggregate,
        results_path=staged_results,
        captured_environment=captured_environment,
    )
    _write_json(staged_outputs / "published_results_manifest.json", published)
    _write_json(staged_outputs / "dashboard_manifest.json", _dashboard_manifest(stage, aggregate))
    verify_publication(root, stage)
    return _json_load(staged_outputs / "dashboard_manifest.json")


def _json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _publication_destinations(stage: Path, root: Path) -> list[tuple[Path, Path]]:
    return [
        (stage / "outputs" / "results.json", root / "outputs" / "results.json"),
        (stage / "outputs" / "dashboard", root / "outputs" / "dashboard"),
        (stage / "docs", root / "docs"),
        (
            stage / "outputs" / "published_results_manifest.json",
            root / "outputs" / "published_results_manifest.json",
        ),
        (
            stage / "outputs" / "dashboard_manifest.json",
            root / "outputs" / "dashboard_manifest.json",
        ),
    ]


def _install_staged_destination(source: Path, destination: Path) -> None:
    """Single commit-phase seam used by rollback failure-injection tests."""
    os.replace(source, destination)


def _commit_publication(
    stage: Path,
    root: Path,
    post_commit_validate: Any | None = None,
) -> None:
    destinations = _publication_destinations(stage, root)
    backup_root = stage / ".publication-backups"
    backup_root.mkdir()
    backups: list[tuple[Path, Path, bool]] = []
    try:
        for index, (_, destination) in enumerate(destinations):
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = backup_root / str(index)
            existed = _path_exists(destination)
            if existed:
                os.replace(destination, backup)
            backups.append((destination, backup, existed))
        for source, destination in destinations:
            _install_staged_destination(source, destination)
        if post_commit_validate is not None:
            post_commit_validate()
    except BaseException:
        for destination, backup, existed in reversed(backups):
            if _path_exists(destination):
                _remove_known_tree(destination)
            if existed and _path_exists(backup):
                os.replace(backup, destination)
        raise
    finally:
        if _path_exists(backup_root):
            _remove_known_tree(backup_root)


def publish_site(
    repository_root: str | Path,
    results_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    results = Path(results_path) if results_path else root / "outputs" / "results.json"
    if not results.is_absolute():
        results = root / results
    _regular_file(results, "Forecast results input")
    canonical_results = root / "outputs" / "results.json"
    resolved_results = results.resolve()
    if resolved_results != canonical_results.resolve():
        protected_files = {
            canonical_results.resolve(),
            (root / "outputs" / "published_results_manifest.json").resolve(),
            (root / "outputs" / "dashboard_manifest.json").resolve(),
        }
        protected_trees = {
            (root / "docs").resolve(),
            (root / "outputs" / "dashboard").resolve(),
        }
        if resolved_results in protected_files or any(
            protected in resolved_results.parents for protected in protected_trees
        ):
            raise RuntimeError(
                "Custom results input must not overlap any publication destination"
            )
    _assert_replace_safe(root / "docs")

    captured_environment = _deterministic_environment(root)
    stage = Path(tempfile.mkdtemp(prefix=".publication-stage-", dir=root))
    try:
        manifest = _build_publication_stage(
            root, stage, results, captured_environment
        )
        _commit_publication(stage, root, lambda: verify_publication(root))
    finally:
        if _path_exists(stage):
            _remove_known_tree(stage)
    return manifest


def check_site(repository_root: str | Path) -> None:
    root = Path(repository_root).resolve()
    _assert_replace_safe(root / "docs")
    results = root / "outputs" / "results.json"
    _regular_file(results, "Canonical forecast results")
    captured_environment = _deterministic_environment(root)
    stage = Path(tempfile.mkdtemp(prefix=".publication-check-", dir=root))
    try:
        _build_publication_stage(
            root, stage, results, captured_environment
        )
        for expected, actual in _publication_destinations(stage, root):
            if expected.is_dir():
                _compare_trees(expected, actual, actual.relative_to(root).as_posix())
            else:
                _compare_files(expected, actual, actual.relative_to(root).as_posix())
        verify_publication(root)
    finally:
        if _path_exists(stage):
            _remove_known_tree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated parity")
    parser.add_argument("--results", help="optional forecast results JSON to publish")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.check:
        if args.results:
            parser.error("--results cannot be combined with --check")
        check_site(root)
        print("Static site and anomaly artifact parity verified.")
    else:
        manifest = publish_site(root, args.results)
        print(f"Published {manifest['product_count']} product artifacts to docs/.")


if __name__ == "__main__":
    main()
