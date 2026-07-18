"""Local presentation dashboard for the anomaly research results.

Serves the authored frontend and optional APIs backed by the exact JSON files
published to GitHub Pages. The server never performs a separate aggregation.

Run (from repo root): uv run python webapp/server.py
or:                   uv run python -m webapp.server
Then open:            http://127.0.0.1:9001
Override port:        VONAVE_ANOMALIE_PORT=9011 uv run python webapp/server.py
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
PUBLISHED_DATA_DIR = ROOT_DIR / "docs" / "data"
RESULTS_PATH = PUBLISHED_DATA_DIR / "results.json"
ANOMALY_DASHBOARD_PATH = PUBLISHED_DATA_DIR / "anomaly-dashboard-v2.json"
ANOMALY_PRODUCTS_DIR = PUBLISHED_DATA_DIR / "anomaly-products-v2"

app = FastAPI(title="Vonave Anomalie Dashboard")


def _read_published_json(path: Path, guidance: str) -> JSONResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=guidance)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Published artifact is unreadable: {path.name}: {exc}",
        ) from exc
    return JSONResponse(payload)


@app.get("/api/results")
def get_results() -> JSONResponse:
    return _read_published_json(
        RESULTS_PATH,
        "outputs/results.json not found. Restore the checked-in published snapshot.",
    )


@app.get("/data/results.json")
def published_results_file() -> FileResponse:
    if not RESULTS_PATH.is_file():
        raise HTTPException(status_code=404, detail="Published forecast results are unavailable.")
    return FileResponse(RESULTS_PATH, media_type="application/json")


@app.get("/data/anomaly-dashboard-v2.json")
def published_anomaly_file() -> FileResponse:
    if not ANOMALY_DASHBOARD_PATH.is_file():
        raise HTTPException(status_code=404, detail="Published anomaly snapshot is unavailable.")
    return FileResponse(ANOMALY_DASHBOARD_PATH, media_type="application/json")


@app.get("/data/anomaly-products-v2/product-{product_id}.json")
def published_anomaly_product_file(product_id: int) -> FileResponse:
    path = ANOMALY_PRODUCTS_DIR / f"product-{product_id}.json"
    if product_id < 1 or product_id > 30 or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Product {product_id} is not published.")
    return FileResponse(path, media_type="application/json")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve the explicit SVG favicon for browsers that still request /favicon.ico."""
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/anomalies")
def anomalies_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/anomaly-lab")
def get_anomaly_lab() -> JSONResponse:
    return _read_published_json(
        ANOMALY_DASHBOARD_PATH,
        "Canonical anomaly artifact not found. Run 'uv run python ml/publish_site.py'.",
    )


@app.get("/api/anomaly-lab/product/{product_id}")
def get_anomaly_product(product_id: int) -> JSONResponse:
    if product_id < 1 or product_id > 30:
        raise HTTPException(status_code=404, detail=f"Product {product_id} is not published.")
    return _read_published_json(
        ANOMALY_PRODUCTS_DIR / f"product-{product_id}.json",
        f"Published anomaly artifact for product {product_id} not found.",
    )


@app.get("/dataset")
def dataset_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dataset.html")


@app.get("/evaluation")
def evaluation_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "evaluation.html")


@app.get("/control")
def control_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "model.html")


@app.get("/model/{slug}")
def model_page(slug: str) -> FileResponse:
    if slug != "neuralnet":
        raise HTTPException(
            status_code=404,
            detail="Only the NeuralNet control is exposed in the anomaly research app.",
        )
    return control_page()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("VONAVE_ANOMALIE_PORT", "9001"))
    uvicorn.run(app, host="127.0.0.1", port=port)
