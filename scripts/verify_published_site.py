"""Verify every checked-in publication artifact using only the standard library."""
from __future__ import annotations

from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SITE_PAGES = ("index.html", "dataset.html", "evaluation.html", "model.html")
PAGE_TITLE = "NOTINO - anomalie"
STRIP_GEOMETRY = {
    "box-sizing": "border-box",
    "width": "100%",
    "max-width": "none",
    "min-height": "216px",
    "margin": "0",
    "padding": "40px 56px",
    "border-bottom": "6px solid var(--mc)",
}
MODEL_HERO_SELECTORS = (
    ".model-hero",
    ".model-hero .model-badge",
    ".model-hero h1",
    ".model-hero p.blurb",
    ".model-hero a.source-link",
    ".model-hero a.source-link:hover",
    ".model-hero",
    ".model-hero code",
)
VOID_HTML_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _fingerprint(path: Path) -> dict[str, int | str]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"Publication entry is not a regular file: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size": info.st_size}


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable publication JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Publication JSON must be an object: {path}")
    return payload


def _regular_files(root: Path) -> dict[str, Path]:
    try:
        root_info = root.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Publication tree is missing: {root}") from exc
    if not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeError(f"Publication tree is not a real directory: {root}")
    files: dict[str, Path] = {}

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise RuntimeError(f"Publication tree contains a symlink: {path}")
                if stat.S_ISDIR(info.st_mode):
                    visit(path)
                elif stat.S_ISREG(info.st_mode):
                    files[path.relative_to(root).as_posix()] = path
                else:
                    raise RuntimeError(f"Publication tree contains a non-regular entry: {path}")

    visit(root)
    return files


def _css_declarations(body: str) -> dict[str, str]:
    return {
        name.strip(): value.strip()
        for declaration in body.split(";")
        if ":" in declaration
        for name, value in [declaration.split(":", 1)]
    }


def _model_hero_rules(css: str) -> list[tuple[str, dict[str, str]]]:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    rules = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector = match.group(1).strip()
        if ".model-hero" not in selector:
            continue
        declarations = _css_declarations(match.group(2))
        rules.append((selector, declarations))
    return rules


def _outer_geometry(declarations: dict[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in declarations.items()
        if (
            name in {
                "box-sizing",
                "width",
                "min-width",
                "max-width",
                "height",
                "min-height",
                "max-height",
                "margin",
                "padding",
                "border-bottom",
            }
            or name.startswith("margin-")
            or name.startswith("padding-")
        )
    }


class _ShellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_body = False
        self.stack: list[str] = []
        self.top_level: list[tuple[str, tuple[str, ...]]] = []
        self.model_heroes: list[tuple[str, tuple[str, ...], dict[str, str | None]]] = []
        self.titles: list[str] = []
        self._title_parts: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        classes = tuple((attributes.get("class") or "").split())
        if tag == "title":
            self._title_parts = []
        if "model-hero" in classes:
            self.model_heroes.append((tag, classes, attributes))
        if tag == "body":
            self.in_body = True
            self.stack.clear()
            return
        if not self.in_body:
            return
        if not self.stack:
            self.top_level.append((tag, classes))
        if tag not in VOID_HTML_ELEMENTS:
            self.stack.append(tag)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        classes = tuple((attributes.get("class") or "").split())
        if "model-hero" in classes:
            self.model_heroes.append((tag, classes, attributes))
        if self.in_body and not self.stack:
            self.top_level.append((tag, classes))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._title_parts is not None:
            self.titles.append("".join(self._title_parts).strip())
            self._title_parts = None
        if tag == "body":
            self.in_body = False
            self.stack.clear()
            return
        if not self.in_body or tag not in self.stack:
            return
        while self.stack:
            if self.stack.pop() == tag:
                break

    def handle_data(self, data: str) -> None:
        if self._title_parts is not None:
            self._title_parts.append(data)


def _verify_page_shell(path: Path) -> None:
    parser = _ShellParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    if parser.titles != [PAGE_TITLE]:
        raise RuntimeError(f"Generated page title contract failed: {path.name}")
    if len(parser.model_heroes) != 1:
        raise RuntimeError(f"Generated description-strip count failed: {path.name}")
    tag, classes, attributes = parser.model_heroes[0]
    if tag != "header" or classes != ("model-hero",):
        raise RuntimeError(f"Generated description-strip identity failed: {path.name}")
    if not re.fullmatch(r"--mc:#[0-9a-fA-F]{6}", attributes.get("style") or ""):
        raise RuntimeError(f"Generated description-strip accent failed: {path.name}")
    try:
        hero_index = parser.top_level.index(("header", ("hero",)))
    except ValueError as exc:
        raise RuntimeError(f"Generated shared hero is missing: {path.name}") from exc
    if parser.top_level[hero_index + 1:hero_index + 2] != [
        ("header", ("model-hero",))
    ]:
        raise RuntimeError(f"Generated description-strip position failed: {path.name}")


def _verify_site_shell(docs: Path) -> None:
    for name in SITE_PAGES:
        _verify_page_shell(docs / name)

    css = (docs / "styles.css").read_text(encoding="utf-8")
    rules = _model_hero_rules(css)
    if tuple(selector for selector, _ in rules) != MODEL_HERO_SELECTORS:
        raise RuntimeError("Shared description-strip selector ownership failed")
    if _outer_geometry(rules[0][1]) != STRIP_GEOMETRY:
        raise RuntimeError("Shared description-strip geometry ownership failed")
    if _outer_geometry(rules[6][1]) != {
        "padding-left": "24px",
        "padding-right": "24px",
    }:
        raise RuntimeError("Shared description-strip responsive geometry failed")
    if "scrollbar-gutter: stable;" not in css:
        raise RuntimeError("Stable scrollbar-gutter contract failed")
    for forbidden in (
        ".page-description-strip",
        ".anomaly-page-hero",
        ".dataset-hero",
        ".evaluation-hero",
    ):
        if forbidden in css:
            raise RuntimeError(f"Page-specific description geometry found: {forbidden}")

    for script in docs.glob("*.js"):
        source = script.read_text(encoding="utf-8")
        if "document.title" in source or "page-title" in source:
            raise RuntimeError(f"Generated script mutates the browser title: {script.name}")


def _require_fingerprint(record: Any, path: Path, label: str) -> None:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} fingerprint is not an object")
    expected = {"sha256": record.get("sha256"), "size": record.get("size")}
    if _fingerprint(path) != expected:
        raise RuntimeError(f"{label} fingerprint mismatch: {path}")


def _source_hash(source_root: Path, names: tuple[str, ...]) -> str:
    records = [
        {"path": name, "sha256": _fingerprint(source_root / name)["sha256"]}
        for name in names
    ]
    return _canonical_hash(sorted(records, key=lambda item: item["path"]))


def _dependency_hash(source_root: Path) -> str:
    records = [
        {"path": name, "sha256": _fingerprint(source_root / name)["sha256"]}
        for name in ("pyproject.toml", "uv.lock")
        if (source_root / name).is_file()
    ]
    return _canonical_hash(records)


def _verify_source_identity(source_root: Path, aggregate: dict[str, Any]) -> None:
    records = aggregate.get("source_inputs")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Anomaly aggregate has no explicit source inputs")
    seen: set[str] = set()
    normalized = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise RuntimeError("Anomaly source-input schema is invalid")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise RuntimeError(f"Unsafe or duplicate anomaly source path: {relative}")
        seen.add(relative.as_posix())
        _require_fingerprint(record, source_root / relative, "Anomaly source")
        normalized.append(record)
    if aggregate.get("source_manifest_hash") != _canonical_hash(normalized):
        raise RuntimeError("Anomaly source_manifest_hash mismatch")
    if aggregate.get("source_hash") != _source_hash(source_root, (
        "webapp/anomaly_dashboard.py",
        "ml/anomaly_detection.py",
        "ml/artifact_provenance.py",
    )):
        raise RuntimeError("Anomaly aggregate source_hash mismatch")
    if aggregate.get("dependency_manifest_hash") != _dependency_hash(source_root):
        raise RuntimeError("Anomaly aggregate dependency hash mismatch")


def _verify_product(
    payload: dict[str, Any],
    *,
    product_id: int,
    source_manifest_hash: str,
) -> None:
    if payload.get("schema_version") != "anomaly-product-v2":
        raise RuntimeError(f"Unexpected product schema for product {product_id}")
    if payload.get("available") is not True or payload.get("product_id") != product_id:
        raise RuntimeError(f"Product identity/availability mismatch for product {product_id}")
    if payload.get("source_manifest_hash") != source_manifest_hash:
        raise RuntimeError(f"Product source_manifest_hash mismatch for product {product_id}")
    if not isinstance(payload.get("summary"), dict):
        raise RuntimeError(f"Product summary is missing for product {product_id}")
    for field in ("timeline", "top_exceedances", "future_context"):
        if not isinstance(payload.get(field), list):
            raise RuntimeError(f"Product {field} is invalid for product {product_id}")


def _verify_forecast_results(path: Path) -> dict[str, Any]:
    payload = _json_object(path)
    required = {
        "config", "models", "selection", "submission", "history",
        "forecasts_by_strategy", "dev_summary_all", "benchmark_summary_all",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"Forecast results schema is missing fields: {missing}")
    config = payload["config"]
    models = payload["models"]
    selection = payload["selection"]
    submission = payload["submission"]
    if (
        not isinstance(config, dict)
        or isinstance(config.get("num_products"), bool)
        or not isinstance(config.get("num_products"), int)
        or config["num_products"] <= 0
        or isinstance(config.get("horizon"), bool)
        or not isinstance(config.get("horizon"), int)
        or config["horizon"] <= 0
        or not isinstance(models, list)
        or not models
        or not isinstance(selection, dict)
        or not isinstance(submission, list)
        or len(submission) != config["num_products"] * config["horizon"]
    ):
        raise RuntimeError("Forecast results schema/count invariants are invalid")
    model_keys = {
        model.get("key") for model in models
        if isinstance(model, dict) and isinstance(model.get("key"), str)
    }
    if len(model_keys) != len(models) or selection.get("canonical_model") not in model_keys:
        raise RuntimeError("Forecast results selected model identity is invalid")
    strategies = payload["forecasts_by_strategy"]
    if (
        not isinstance(strategies, dict)
        or selection.get("canonical_strategy") not in strategies
    ):
        raise RuntimeError("Forecast results selected strategy identity is invalid")
    return payload


def verify_publication(
    source_root: str | Path,
    publication_root: str | Path | None = None,
) -> None:
    """Validate the complete canonical publication set.

    ``source_root`` supplies checked-in source inputs. ``publication_root`` may
    point at a fully populated staging tree before any destination is replaced.
    """
    source = Path(source_root).resolve()
    publication = Path(publication_root).resolve() if publication_root else source
    docs = publication / "docs"
    dashboard = publication / "outputs" / "dashboard"
    docs_files = _regular_files(docs)
    dashboard_files = _regular_files(dashboard)

    results = publication / "outputs" / "results.json"
    published_path = publication / "outputs" / "published_results_manifest.json"
    dashboard_manifest_path = publication / "outputs" / "dashboard_manifest.json"
    for path in (results, published_path, dashboard_manifest_path):
        _fingerprint(path)
    _verify_forecast_results(results)

    site_manifest = _json_object(docs / "site-manifest.json")
    if site_manifest.get("schema_version") != "static-dashboard-site-v2":
        raise RuntimeError("Unexpected docs/site-manifest.json schema")
    rows = site_manifest.get("files")
    if not isinstance(rows, list):
        raise RuntimeError("Site manifest files must be a list")
    site_records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            raise RuntimeError("Site manifest file schema is invalid")
        relative = Path(row["path"])
        name = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or name in site_records:
            raise RuntimeError(f"Unsafe or duplicate site-manifest path: {name}")
        site_records[name] = row
    actual_site = set(docs_files) - {"site-manifest.json"}
    if actual_site != set(site_records):
        raise RuntimeError(
            f"Generated-site ownership mismatch: missing={sorted(set(site_records) - actual_site)}, "
            f"extra={sorted(actual_site - set(site_records))}"
        )
    for relative, record in site_records.items():
        _require_fingerprint(record, docs / relative, "Generated-site")
    _verify_site_shell(docs)

    aggregate_path = dashboard / "anomaly-dashboard-v2.json"
    aggregate = _json_object(aggregate_path)
    if aggregate.get("schema_version") != "anomaly-dashboard-v3":
        raise RuntimeError("Unexpected anomaly aggregate schema")
    source_manifest_hash = aggregate.get("source_manifest_hash")
    if not isinstance(source_manifest_hash, str) or len(source_manifest_hash) != 64:
        raise RuntimeError("Anomaly aggregate source_manifest_hash is invalid")
    _verify_source_identity(source, aggregate)
    if (
        site_manifest.get("source_manifest_hash") != source_manifest_hash
        or site_manifest.get("generated_at") != aggregate.get("generated_at")
        or site_manifest.get("generated_at_basis") != aggregate.get("generated_at_basis")
    ):
        raise RuntimeError("Site manifest snapshot/source identity mismatch")
    products = aggregate.get("audit", {}).get("products")
    if products != list(range(1, 31)):
        raise RuntimeError(f"Expected aggregate products 1..30, found {products}")

    expected_dashboard = {"anomaly-dashboard-v2.json"} | {
        f"anomaly-products-v2/product-{product_id}.json" for product_id in products
    }
    if set(dashboard_files) != expected_dashboard:
        raise RuntimeError("outputs/dashboard has an unexpected file set")
    for product_id in products:
        relative = f"anomaly-products-v2/product-{product_id}.json"
        _verify_product(
            _json_object(dashboard / relative),
            product_id=product_id,
            source_manifest_hash=source_manifest_hash,
        )

    parity = {
        "data/results.json": results,
        "data/anomaly-dashboard-v2.json": aggregate_path,
        "data/published_results_manifest.json": published_path,
    }
    parity.update({
        f"data/{relative}": path for relative, path in dashboard_files.items()
        if relative.startswith("anomaly-products-v2/")
    })
    for relative, canonical in parity.items():
        if _fingerprint(docs / relative) != _fingerprint(canonical):
            raise RuntimeError(f"Published docs data parity mismatch: {relative}")

    published = _json_object(published_path)
    if published.get("schema_version") != "published-results-v4":
        raise RuntimeError("Unexpected published-results manifest schema")
    for field in (
        "artifact_schema_version", "source_manifest_hash", "source_hash",
        "input_data_hash", "dependency_manifest_hash", "forecast_results",
        "outputs", "site_artifacts", "data_hashes",
    ):
        if field not in published:
            raise RuntimeError(f"Published-results manifest is missing {field}")
    if (
        published["artifact_schema_version"] != aggregate["schema_version"]
        or published["source_manifest_hash"] != source_manifest_hash
        or published.get("generated_at") != aggregate.get("generated_at")
        or published.get("generated_at_basis") != aggregate.get("generated_at_basis")
        or published.get("snapshot_as_of") != aggregate.get("snapshot_as_of")
    ):
        raise RuntimeError("Published-results aggregate identity mismatch")
    expected_data_hashes = {
        row["path"]: row["sha256"] for row in aggregate["source_inputs"]
    }
    if published["data_hashes"] != expected_data_hashes:
        raise RuntimeError("Published-results data hashes mismatch")
    if published["input_data_hash"] != _canonical_hash(expected_data_hashes):
        raise RuntimeError("Published-results input_data_hash mismatch")
    if published["configuration_hash"] != _canonical_hash({
        "recommendation": aggregate["recommendation"],
        "research_status": aggregate["research_status"],
    }):
        raise RuntimeError("Published-results configuration hash mismatch")
    if published["source_hash"] != _source_hash(source, (
        "webapp/anomaly_dashboard.py",
        "ml/publish_site.py",
        "ml/anomaly_detection.py",
        "ml/artifact_provenance.py",
    )):
        raise RuntimeError("Published-results source hash mismatch")
    if published["dependency_manifest_hash"] != _dependency_hash(source):
        raise RuntimeError("Published-results dependency hash mismatch")
    _require_fingerprint(published["forecast_results"], results, "Forecast results")
    if published["forecast_results"].get("path") != "outputs/results.json":
        raise RuntimeError("Published-results forecast path is invalid")
    expected_outputs = {
        relative: _fingerprint(path) for relative, path in sorted(dashboard_files.items())
    }
    if published["outputs"] != expected_outputs:
        raise RuntimeError("Published-results output fingerprints mismatch")
    expected_site_artifacts = {
        relative: record["sha256"]
        for relative, record in site_records.items()
        if relative not in {"data/published_results_manifest.json"}
    }
    if published["site_artifacts"] != expected_site_artifacts:
        raise RuntimeError("Published-results site artifact hashes mismatch")

    dashboard_manifest = _json_object(dashboard_manifest_path)
    if dashboard_manifest.get("schema_version") != "dashboard-publication-v2":
        raise RuntimeError("Unexpected dashboard publication schema")
    if (
        dashboard_manifest.get("product_count") != len(products)
        or dashboard_manifest.get("source_manifest_hash") != source_manifest_hash
    ):
        raise RuntimeError("Dashboard manifest count/source identity mismatch")
    for field, expected_path in (
        ("forecast_results", "outputs/results.json"),
        ("anomaly_aggregate", "outputs/dashboard/anomaly-dashboard-v2.json"),
        ("site_manifest", "docs/site-manifest.json"),
        ("published_results_manifest", "outputs/published_results_manifest.json"),
    ):
        record = dashboard_manifest.get(field)
        if not isinstance(record, dict) or record.get("path") != expected_path:
            raise RuntimeError(f"Dashboard manifest {field} path mismatch")
        _require_fingerprint(record, publication / expected_path, "Dashboard manifest")
    product_record = dashboard_manifest.get("anomaly_products")
    expected_product_files = {
        f"outputs/dashboard/{relative}": _fingerprint(path)
        for relative, path in sorted(dashboard_files.items())
        if relative.startswith("anomaly-products-v2/")
    }
    if (
        not isinstance(product_record, dict)
        or product_record.get("count") != len(products)
        or product_record.get("files") != expected_product_files
        or product_record.get("sha256") != _canonical_hash(expected_product_files)
    ):
        raise RuntimeError("Dashboard product manifest mismatch")


def main() -> None:
    verify_publication(ROOT)
    print("Checked-in Pages artifact integrity verified.")


if __name__ == "__main__":
    main()
