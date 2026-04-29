"""
Integration tests — FastAPI REST entry point (POST /lifting/depth-estimation).

Binary artefacts (depth_image, point_cloud_ply) are returned as base64-encoded
strings inside the JSON response and decoded back to bytes before being saved.

Each group clears the in-memory model cache before run 1 to guarantee a cold
start, then verifies that in-memory caching accelerates runs 2-N.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("open3d", reason="open3d required — run: uv python pin 3.12 && uv sync")

from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_BACKEND   # noqa: E402
from vizion3d.lifting.handlers import DepthEstimationHandler         # noqa: E402
from vizion3d.server.rest.app import app                             # noqa: E402

N_RUNS = 5

client = TestClient(app, raise_server_exceptions=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_outputs(data: dict, run_dir: Path, run: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(run_dir / f"run_{run:02d}")

    (Path(prefix + ".depth_map.json")).write_text(json.dumps(data["depth_map"]))

    if data.get("depth_image"):
        Path(prefix + "_depth.png").write_bytes(base64.b64decode(data["depth_image"]))

    if data.get("point_cloud_ply"):
        Path(prefix + "_point_cloud.ply").write_bytes(
            base64.b64decode(data["point_cloud_ply"])
        )


def _run_group(
    model_backend: str,
    indoor_image_bytes: bytes,
    run_dir: Path,
    scenario: str,
    timing_collector,
) -> list[float]:
    DepthEstimationHandler._depth_anything_models.clear()

    timings: list[float] = []

    for run in range(1, N_RUNS + 1):
        t0 = time.perf_counter()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("indoor_scene.jpg", indoor_image_bytes, "image/jpeg")},
            data={
                "model_backend": model_backend,
                "return_depth_image": "true",
                "return_point_cloud": "true",
            },
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        assert response.status_code == 200, (
            f"Run {run} failed: {response.status_code} — {response.text[:300]}"
        )

        data = response.json()
        _save_outputs(data, run_dir, run)
        timing_collector.add("REST", scenario, run, elapsed, str(run_dir))

        # ── per-run assertions ────────────────────────────────────────────────
        assert isinstance(data["depth_map"], list) and len(data["depth_map"]) > 0
        assert data["max_depth"] > data["min_depth"], \
            "Depth variation expected for a real indoor scene"
        assert data["depth_image"] is not None, \
            "depth_image was requested but missing from response"
        assert data["point_cloud_ply"] is not None, \
            "point_cloud_ply was requested but missing from response"

        # Binary fields should decode to valid PLY / PNG
        png_bytes = base64.b64decode(data["depth_image"])
        assert png_bytes[:4] == b"\x89PNG", "depth_image is not a valid PNG"

        ply_bytes = base64.b64decode(data["point_cloud_ply"])
        assert ply_bytes.startswith(b"ply\n"), "point_cloud_ply is not a valid PLY"

    assert len(DepthEstimationHandler._depth_anything_models) > 0, \
        "Model should be cached in memory after first REST inference"

    assert timings[0] < 10.0, (
        f"[REST / {scenario}] "
        f"Cold load took {timings[0]:.3f}s — expected < 10s"
    )
    for i, t in enumerate(timings[1:], start=2):
        assert t < 1.0, (
            f"[REST / {scenario}] "
            f"Run {i} took {t:.3f}s — expected < 1s (model should be cached in memory)"
        )

    return timings


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_rest_default_model(indoor_image_bytes, tmp_path, timing_collector):
    """5 POST requests with the default model backend."""
    _run_group(
        model_backend=DEFAULT_DEPTH_MODEL_BACKEND,
        indoor_image_bytes=indoor_image_bytes,
        run_dir=tmp_path / "rest_default",
        scenario="Default model",
        timing_collector=timing_collector,
    )


def test_rest_local_model(
    indoor_image_bytes, local_model_path, tmp_path, timing_collector
):
    """5 POST requests with an explicit local .pth path."""
    _run_group(
        model_backend=local_model_path,
        indoor_image_bytes=indoor_image_bytes,
        run_dir=tmp_path / "rest_local",
        scenario="Local model",
        timing_collector=timing_collector,
    )
