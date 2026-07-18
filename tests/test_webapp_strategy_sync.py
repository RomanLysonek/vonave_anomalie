import json
from pathlib import Path
import shutil

import pytest
from fastapi import HTTPException

from scripts.verify_published_site import (
    PAGE_TITLE,
    _css_rules,
    _verify_page_shell,
    _verify_site_shell,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "webapp" / "static"
DOCS = ROOT / "docs"
PAGES = ("index.html", "dataset.html", "evaluation.html", "model.html")
FORBIDDEN_PORTAL_TEXT = (
    "suite" + "-shell",
    "Classical" + " Forecasting",
    "Chronos-2" + " Challenger",
)


def test_root_is_the_anomaly_research_experience() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert f"<title>{PAGE_TITLE}</title>" in html
    assert "DAVID / DBAAS knowledge transfer" in html
    assert "Causal statistical scores" in html
    assert "Portfolio &amp; V2 diagnostics" in html
    assert "Future-context novelty" in html
    assert "Bounded experiments" in html
    assert "No adjudicated anomaly labels exist" in html
    assert "anomaly_mode=off" in html
    assert 'id="strategy-select"' not in html
    assert "Model comparison" not in html


@pytest.mark.parametrize("directory", (STATIC, DOCS), ids=("authored", "generated"))
def test_every_page_uses_one_shared_description_strip_and_constant_title(
    directory: Path,
) -> None:
    _verify_site_shell(directory)


def test_shell_verifier_rejects_alternate_title_and_strip_markup(
    tmp_path: Path,
) -> None:
    valid = f"""
      <html><head><title>{PAGE_TITLE}</title></head><body>
      <header class="hero"></header>
      <header class="description-strip model-hero" style="--mc:#b76600"></header>
      </body></html>
    """
    for invalid in (
        valid.replace("<head>", "<head><title>Other</title>"),
        valid.replace(
            'class="description-strip model-hero"',
            'class="description-strip model-hero extra"',
        ),
    ):
        path = tmp_path / "page.html"
        path.write_text(invalid, encoding="utf-8")
        with pytest.raises(RuntimeError):
            _verify_page_shell(path)


def test_shell_verifier_detects_complex_selector_geometry_override() -> None:
    css = '.description-strip:not(.unused .selector) { margin-inline: 4px; }'
    assert _css_rules(css, ".description-strip")[0][0] == (
        ".description-strip:not(.unused .selector)"
    )


def test_shell_verifier_requires_responsive_override_media_scope(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site"
    shutil.copytree(STATIC, site)
    styles = site / "styles.css"
    original = styles.read_text(encoding="utf-8")
    css = original.replace(
        "@media (max-width: 900px) {\n  :root { --page-padding-inline: 24px; }",
        ":root { --page-padding-inline: 24px; }\n@media (max-width: 900px) {",
        1,
    )
    styles.write_text(css, encoding="utf-8")
    with pytest.raises(RuntimeError, match="media-query scope"):
        _verify_site_shell(site)

    css = original.replace(
        "@media (max-width: 900px) {",
        "@media (min-width: 901px) {",
        1,
    )
    css += (
        "\n/* @media (max-width: 900px) {\n"
        "  :root { --page-padding-inline: 24px; }\n} */\n"
    )
    styles.write_text(css, encoding="utf-8")
    with pytest.raises(RuntimeError, match="media-query scope"):
        _verify_site_shell(site)


def test_authored_pages_have_independent_anomaly_navigation() -> None:
    common = (STATIC / "common.js").read_text(encoding="utf-8")
    expected = (
        'label: "Anomaly overview"',
        'label: "Data Story"',
        'label: "Evaluation"',
        'label: "Control forecast"',
    )
    for label in expected:
        assert label in common

    for name in PAGES:
        html = (STATIC / name).read_text(encoding="utf-8")
        assert '<html lang="en-GB">' in html
        assert 'aria-label="Anomaly dashboard sections"' in html
        assert "NOTINO" in html
        for forbidden in FORBIDDEN_PORTAL_TEXT:
            assert forbidden not in html

    authored_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in STATIC.iterdir()
        if path.is_file()
    )
    for forbidden in FORBIDDEN_PORTAL_TEXT:
        assert forbidden not in authored_text


def test_supporting_pages_keep_anomaly_scope() -> None:
    transfer = (STATIC / "dataset.html").read_text(encoding="utf-8")
    assert "From DAVID / DBAAS to this assignment" in transfer
    assert "Ranking" in transfer
    assert "No detector accuracy" in transfer
    assert "false-positive rate" in transfer

    evaluation = (STATIC / "evaluation.html").read_text(encoding="utf-8")
    assert "Leakage-safe experiment contract" in evaluation
    assert "Origin-known anomaly state" in evaluation
    assert "Bounded robust fitting" in evaluation
    assert "V2 only when verified" in evaluation
    assert "Control / <code>anomaly_mode=off</code> remains recommended" in evaluation

    control = (STATIC / "model.html").read_text(encoding="utf-8")
    assert "Quantity Forecast Dashboard" in control
    assert "Why this control exists" in control
    assert 'id="strategy-select"' not in control


def test_server_routes_point_to_anomaly_first_pages() -> None:
    from webapp import server

    assert Path(server.index().path) == STATIC / "index.html"
    assert Path(server.anomalies_page().path) == STATIC / "index.html"
    assert Path(server.dataset_page().path) == STATIC / "dataset.html"
    assert Path(server.evaluation_page().path) == STATIC / "evaluation.html"
    assert Path(server.control_page().path) == STATIC / "model.html"
    assert Path(server.model_page("neuralnet").path) == STATIC / "model.html"
    assert Path(server.favicon().path) == STATIC / "favicon.svg"

    for slug in ("xgboost", "lightgbm", "ensemble", "seasonalnaive", "movingavg28"):
        with pytest.raises(HTTPException) as exc_info:
            server.model_page(slug)
        assert exc_info.value.status_code == 404


def test_static_control_ignores_model_query_and_has_no_model_directory() -> None:
    model_script = (STATIC / "model.js").read_text(encoding="utf-8")
    common_script = (STATIC / "common.js").read_text(encoding="utf-8")

    assert 'function currentSlug() {\n  return "neuralnet";\n}' in model_script
    assert "URLSearchParams" not in model_script
    assert "renderNotFound" not in model_script
    assert "modelHref(" not in model_script
    assert 'href: controlHref()' in common_script
    assert "model.html?model=" not in common_script
    assert 'return slug === "neuralnet" ? controlHref() : "#";' in common_script
    assert "data.models || []" not in common_script.split("function renderNav", 1)[1]


def test_static_and_api_anomaly_artifacts_are_identical() -> None:
    from webapp import server

    aggregate = json.loads((DOCS / "data" / "anomaly-dashboard-v2.json").read_text())
    assert json.loads(server.get_anomaly_lab().body) == aggregate

    for product_id in (1, 17, 30):
        static_payload = json.loads(
            (DOCS / "data" / "anomaly-products-v2" / f"product-{product_id}.json").read_text()
        )
        assert json.loads(server.get_anomaly_product(product_id).body) == static_payload


def test_generated_docs_match_authored_identity_and_use_relative_paths() -> None:
    for name in PAGES:
        html = (DOCS / name).read_text(encoding="utf-8")
        assert 'href="./styles.css' in html
        assert 'src="./common.js' in html
        for forbidden in FORBIDDEN_PORTAL_TEXT:
            assert forbidden not in html

    root = (DOCS / "index.html").read_text(encoding="utf-8")
    assert "DAVID / DBAAS knowledge transfer" in root
    assert "window.STATIC_DASHBOARD = true" in root
    script = (DOCS / "anomalies.js").read_text(encoding="utf-8")
    assert "./data/anomaly-dashboard-v2.json" in script
    assert "const relative = `data/anomaly-products-v2/product-" in script
    assert "window.STATIC_DASHBOARD ? `./${relative}`" in script
