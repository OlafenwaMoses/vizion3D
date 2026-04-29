"""
Integration tests — gRPC entry point (live in-process server).

A real gRPC server is started once per session on a random port.
Binary artefacts (depth_image, point_cloud_ply) come back as raw bytes
inside the protobuf response and are written directly to the run directory.

Each group clears the in-memory model cache before run 1 to guarantee a cold
start, then verifies that in-memory caching accelerates runs 2-N.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("open3d", reason="open3d required — run: uv python pin 3.12 && uv sync")

from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_BACKEND   # noqa: E402
from vizion3d.lifting.handlers import DepthEstimationHandler         # noqa: E402
from vizion3d.proto import lifting_pb2                               # noqa: E402

N_RUNS = 5


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_outputs(response, run_dir: Path, run: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(run_dir / f"run_{run:02d}")

    depth_map = [list(row.values) for row in response.depth_map]
    Path(prefix + ".depth_map.json").write_text(json.dumps(depth_map))

    if response.depth_image:
        Path(prefix + "_depth.png").write_bytes(response.depth_image)

    if response.point_cloud_ply:
        Path(prefix + "_point_cloud.ply").write_bytes(response.point_cloud_ply)


def _run_group(
    model_backend: str,
    indoor_image_bytes: bytes,
    run_dir: Path,
    scenario: str,
    grpc_client_stub,
    timing_collector,
) -> list[float]:
    # The gRPC server runs in-process (background thread) — the class-level
    # model cache is shared between the test thread and the server threads.
    DepthEstimationHandler._depth_anything_models.clear()

    timings: list[float] = []

    for run in range(1, N_RUNS + 1):
        request = lifting_pb2.DepthEstimationRequest(
            image_bytes=indoor_image_bytes,
            model_backend=model_backend,
            return_depth_image=True,
            return_point_cloud=True,
        )

        t0       = time.perf_counter()
        response = grpc_client_stub.RunDepthEstimation(request)
        elapsed  = time.perf_counter() - t0
        timings.append(elapsed)

        _save_outputs(response, run_dir, run)
        timing_collector.add("gRPC", scenario, run, elapsed, str(run_dir))

        # ── per-run assertions ────────────────────────────────────────────────
        assert len(response.depth_map) > 0, "depth_map rows are missing"
        assert response.max_depth > response.min_depth, \
            "Depth variation expected for a real indoor scene"
        assert len(response.depth_image) > 0, \
            "depth_image was requested but empty in response"
        assert len(response.point_cloud_ply) > 0, \
            "point_cloud_ply was requested but empty in response"

        # Validate file-format magic bytes
        assert response.depth_image[:4] == b"\x89PNG", \
            "depth_image is not a valid PNG"
        assert response.point_cloud_ply.startswith(b"ply\n"), \
            "point_cloud_ply is not a valid PLY"

    assert len(DepthEstimationHandler._depth_anything_models) > 0, \
        "Model should be cached in memory after first gRPC inference"

    assert timings[1] < timings[0], (
        f"[gRPC / {scenario}] "
        f"Warm inference ({timings[1]:.3f}s) should be faster than cold load "
        f"({timings[0]:.3f}s)."
    )

    return timings


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_grpc_default_model(
    indoor_image_bytes, grpc_client_stub, tmp_path, timing_collector
):
    """5 RPC calls with the default model backend."""
    _run_group(
        model_backend=DEFAULT_DEPTH_MODEL_BACKEND,
        indoor_image_bytes=indoor_image_bytes,
        run_dir=tmp_path / "grpc_default",
        scenario="Default model",
        grpc_client_stub=grpc_client_stub,
        timing_collector=timing_collector,
    )


def test_grpc_local_model(
    indoor_image_bytes, local_model_path, grpc_client_stub, tmp_path, timing_collector
):
    """5 RPC calls with an explicit local .pth path."""
    _run_group(
        model_backend=local_model_path,
        indoor_image_bytes=indoor_image_bytes,
        run_dir=tmp_path / "grpc_local",
        scenario="Local model",
        grpc_client_stub=grpc_client_stub,
        timing_collector=timing_collector,
    )
