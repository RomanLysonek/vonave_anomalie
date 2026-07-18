from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_COMMANDS = (
    ("uv", "run", "python", "webapp/server.py"),
    ("uv", "run", "python", "-m", "webapp.server"),
)
SMOKE_PATHS = (
    "/",
    "/data/anomaly-dashboard-v2.json",
    "/data/anomaly-products-v2/product-1.json",
    "/dataset",
    "/evaluation",
    "/control",
    "/model/neuralnet",
    "/api/results",
    "/api/anomaly-lab",
    "/api/anomaly-lab/product/1",
)
BLOCKED_MODEL_PATHS = (
    "/model/xgboost",
    "/model/lightgbm",
    "/model/ensemble",
    "/model/seasonalnaive",
    "/model/movingavg28",
)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_http(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"server exited with {process.returncode}: {stderr}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(0.1)
    raise AssertionError("server did not become ready")


@pytest.mark.parametrize("command", LAUNCH_COMMANDS)
def test_documented_launch_commands_and_http_surface(command: tuple[str, ...]) -> None:
    port = _free_port()
    env = {**os.environ, "VONAVE_ANOMALIE_PORT": str(port)}
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_http(port, process)
        for path in SMOKE_PATHS:
            with urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
                assert response.status == 200, path
        for path in BLOCKED_MODEL_PATHS:
            with pytest.raises(HTTPError) as exc_info:
                urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)
            assert exc_info.value.code == 404, path
        with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            assert b"DAVID / DBAAS knowledge transfer" in response.read()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
