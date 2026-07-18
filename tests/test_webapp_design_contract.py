import re
from pathlib import Path

import pytest

from scripts.verify_published_site import _css_rules


ROOT = Path(__file__).resolve().parents[1]
AUTHORED = ROOT / "webapp" / "static"
GENERATED = ROOT / "docs"
PAGES = ("index.html", "dataset.html", "evaluation.html", "model.html")

EXPECTED_HEADER = (
    '<header class="hero"> <div class="hero-top"> <div class="brand"> '
    '<span class="brand-logo">NOTINO</span> '
    '<span class="brand-tagline anomaly-brand-tagline">ANOMALIE</span> '
    '</div> <div class="hero-title-block"> '
    '<h1>Quantity Forecast Dashboard</h1> </div> </div> '
    '<nav class="site-nav" id="site-nav" '
    'aria-label="Anomaly dashboard sections"></nav> </header>'
)

EXPECTED_RULES = {
    "header.hero": {
        "position": "relative",
        "z-index": "1",
        "background": "var(--ink)",
        "padding": "28px var(--page-padding-inline) 0",
    },
    ".hero-top": {
        "display": "flex",
        "justify-content": "space-between",
        "align-items": "flex-start",
        "flex-wrap": "wrap",
        "gap": "24px",
        "padding-bottom": "24px",
    },
    ".brand": {
        "display": "flex",
        "align-items": "baseline",
        "gap": "10px",
        "flex-wrap": "wrap",
    },
    ".brand-logo": {
        "font-size": "34px",
        "font-weight": "700",
        "letter-spacing": "0.06em",
        "color": "#ffffff",
    },
    ".brand-tagline": {
        "font-size": "12.5px",
        "font-weight": "500",
        "color": "#b8b8b8",
        "text-transform": "uppercase",
        "letter-spacing": "0.06em",
    },
    ".hero-title-block": {"text-align": "right"},
    ".hero-title-block h1": {
        "margin": "0",
        "font-size": "19px",
        "font-weight": "400",
        "letter-spacing": "0",
        "text-transform": "none",
        "color": "#c8c8c8",
    },
    "nav.site-nav": {
        "position": "relative",
        "z-index": "2",
        "display": "flex",
        "flex-wrap": "wrap",
        "gap": "0",
        "border-top": "1px solid #2a2a2a",
    },
    ".nav-pill": {
        "--pill-color": "#ffffff",
        "display": "inline-flex",
        "align-items": "center",
        "gap": "8px",
        "padding": "14px 20px",
        "border-radius": "0",
        "border": "none",
        "border-right": "1px solid #2a2a2a",
        "background": "transparent",
        "color": "#b8b8b8",
        "font-size": "12.5px",
        "font-weight": "700",
        "text-transform": "uppercase",
        "letter-spacing": "0.05em",
        "text-decoration": "none",
        "transition": "background 0.15s, color 0.15s",
    },
    ".nav-pill::before": {
        "content": '""',
        "width": "9px",
        "height": "9px",
        "border-radius": "0",
        "background": "var(--pill-color)",
        "outline": "1px solid rgba(255, 255, 255, 0.5)",
        "flex": "none",
    },
    ".nav-pill:hover": {"color": "#fff", "background": "#1c1c1c"},
    ".nav-pill.active": {"color": "var(--ink)", "background": "#fff"},
    ".nav-pill.active::before": {"outline": "1px solid var(--ink)"},
}


@pytest.mark.parametrize("directory", (AUTHORED, GENERATED), ids=("authored", "generated"))
def test_every_page_uses_prediction_header_structure(directory: Path) -> None:
    for page in PAGES:
        html = (directory / page).read_text(encoding="utf-8")
        headers = re.findall(r'<header class="hero">.*?</header>', html, re.DOTALL)
        assert len(headers) == 1, f"{directory.name}/{page}"
        assert re.sub(r"\s+", " ", headers[0]).strip() == EXPECTED_HEADER
        assert html.count("<title>NOTINO - anomalie</title>") == 1


def test_header_navigation_geometry_matches_prediction_template() -> None:
    css = (AUTHORED / "styles.css").read_text(encoding="utf-8")
    for selector, expected in EXPECTED_RULES.items():
        rules = [
            declarations
            for actual_selector, declarations in _css_rules(css, selector)
            if actual_selector == selector
        ]
        if selector == ".hero-title-block":
            assert rules == [{"text-align": "right"}, {"text-align": "left"}]
        else:
            assert rules == [expected], selector

    assert "@media (max-width: 900px) {\n  :root { --page-padding-inline: 24px; }" in css
    assert (
        "@media (max-width: 900px) {\n"
        "  :root { --page-padding-inline: 24px; }\n"
        "  .kpi-grid { grid-template-columns: repeat(2, 1fr); }\n"
        "  .panel-grid { grid-template-columns: 1fr; }\n"
        "  .panel-grid .panel + .panel { border-left: 1px solid var(--ink); "
        "border-top: none; margin-top: -1px; }\n"
        "  .hero-title-block { text-align: left; }\n"
        "}"
    ) in css
    assert ".promo-bar, header.hero, main#app, footer p" not in css
    assert "header.hero {\n  width:" not in css
    assert ".anomaly-brand-tagline {\n  color: #f59e0b;" in css


def test_navigation_inventory_and_page_accents_match_contract() -> None:
    common = (AUTHORED / "common.js").read_text(encoding="utf-8")
    expected = (
        ('Anomaly overview', "#f59e0b"),
        ("Data Story", "#a78bfa"),
        ("Evaluation", "#9ca3af"),
        ("Control forecast", "#ffffff"),
    )
    for label, color in expected:
        assert f'label: "{label}", color: "{color}"' in common
    assert common.count("label: ") == len(expected)
    assert "Data & transfer" not in common

    dataset = (AUTHORED / "dataset.html").read_text(encoding="utf-8")
    evaluation = (AUTHORED / "evaluation.html").read_text(encoding="utf-8")
    assert '<header class="description-strip model-hero" style="--mc:#7c3aed">' in dataset
    assert '<header class="description-strip model-hero" style="--mc:#4b5563">' in evaluation
