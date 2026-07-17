"""Local presentation dashboard for the Notino forecast results.

Serves the static frontend (webapp/static/) and a small JSON API that reads
outputs/results.json fresh on every request -- rerun the ML pipeline
(uv run python ml/pipeline.py) or the lightweight
`uv run python ml/export_results.py`, then just refresh the browser.

Run (from repo root): uv run python webapp/server.py
Then open:            http://127.0.0.1:9001
Override port:        VONAVE_ANOMALIE_PORT=9011 uv run python webapp/server.py
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from anomaly_dashboard import (
    build_anomaly_dashboard,
    build_anomaly_status,
    build_product_payload,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_PATH = ROOT_DIR / "outputs" / "results.json"

app = FastAPI(title="Vonave Anomalie Dashboard")


@app.get("/api/results")
def get_results() -> JSONResponse:
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "outputs/results.json not found. Run "
                "'uv run python ml/pipeline.py' (or ml/export_results.py) first."
            ),
        )
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve the explicit SVG favicon for browsers that still request /favicon.ico."""
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/anomalies")
def anomalies_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "anomalies.html")


@app.get("/api/anomaly-lab")
def get_anomaly_lab() -> JSONResponse:
    return JSONResponse(build_anomaly_dashboard(ROOT_DIR))


@app.get("/api/anomaly-lab/status")
def get_anomaly_lab_status() -> JSONResponse:
    return JSONResponse(build_anomaly_status(ROOT_DIR))


@app.get("/api/anomaly-lab/product/{product_id}")
def get_anomaly_product(product_id: int) -> JSONResponse:
    payload = build_product_payload(ROOT_DIR, product_id)
    if not payload.get("available"):
        raise HTTPException(status_code=404, detail=payload.get("message", "Product not found"))
    return JSONResponse(payload)


@app.get("/dataset")
def dataset_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dataset.html")


@app.get("/evaluation")
def evaluation_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "evaluation.html")


@app.get("/model/{slug}")
def model_page(slug: str) -> FileResponse:
    # One shared template; model.js reads `slug` from the URL itself and
    # renders that model's data/colors. Unknown slugs still get the page --
    # model.js shows a clear "not found" state rather than a 404.
    return FileResponse(STATIC_DIR / "model.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("VONAVE_ANOMALIE_PORT", "9001"))
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
    )
