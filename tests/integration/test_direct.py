"""
Integration tests — direct Python command-bus entry point (DepthEstimation.run).

Each test group clears the in-memory model cache before the first inference to
simulate a cold start, then runs N_RUNS iterations.  We assert:
  - every run produces a valid depth map with real depth variation
  - depth image and point cloud are returned and non-trivial
  - all artefacts are persisted to a session tmp directory
  - the model is cached in memory after run 1 (subsequent runs faster)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

pytest.importorskip("open3d", reason="open3d required — run: uv python pin 3.12 && uv sync")

from vizion3d.lifting import DepthEstimation, DepthEstimationCommand  # noqa: E402
from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_URL  # noqa: E402
from vizion3d.lifting.handlers import DepthEstimationHandler  # noqa: E402
from vizion3d.lifting.utils import create_ply_binary  # noqa: E402

N_RUNS = 5
COLD_LIMIT = float(os.environ.get("VIZION3D_TEST_COLD_LIMIT", "10.0"))
WARM_LIMIT = float(os.environ.get("VIZION3D_TEST_WARM_LIMIT", "1.0"))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_outputs(result, run_dir: Path, run: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_dir / f"run_{run:02d}"

    (prefix.with_suffix(".depth_map.json")).write_text(
        json.dumps(result.depth_map)
    )

    if result.depth_image is not None:
        arr = np.asarray(result.depth_image)          # uint16 (H, W)
        PILImage.fromarray(arr).save(str(prefix) + "_depth.png")

    if result.point_cloud is not None and result.point_cloud.has_points():
        pts  = np.asarray(result.point_cloud.points).astype(np.float32)
        cols = (np.asarray(result.point_cloud.colors) * 255).astype(np.uint8)
        ply  = create_ply_binary(pts, cols)
        (str(prefix) + "_point_cloud.ply")
        Path(str(prefix) + "_point_cloud.ply").write_bytes(ply)


def _run_group(
    model_backend: str,
    indoor_image_bytes: bytes,
    run_dir: Path,
    entry_point: str,
    scenario: str,
    timing_collector,
) -> list[float]:
    # Simulate cold start: evict any in-memory loaded model for this path
    DepthEstimationHandler._depth_anything_models.clear()

    timings: list[float] = []

    for run in range(1, N_RUNS + 1):
        t0     = time.perf_counter()
        result = DepthEstimation().run(
            DepthEstimationCommand(
                image_input=indoor_image_bytes,
                model_backend=model_backend,
                return_depth_image=True,
                return_point_cloud=True,
            )
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        _save_outputs(result, run_dir, run)
        timing_collector.add(entry_point, scenario, run, elapsed, str(run_dir))

        # ── per-run assertions ────────────────────────────────────────────────
        assert isinstance(result.depth_map, list) and len(result.depth_map) > 0
        assert all(isinstance(row, list) for row in result.depth_map)
        assert result.max_depth > result.min_depth, \
            "Depth variation expected for a real indoor scene"
        assert result.depth_image is not None, "depth_image was requested but missing"
        assert result.point_cloud is not None and result.point_cloud.has_points(), \
            "point_cloud was requested but missing or empty"

    # ── caching assertion: model must be in memory after run 1 ───────────────
    assert len(DepthEstimationHandler._depth_anything_models) > 0, \
        "Model should be in DepthEstimationHandler._depth_anything_models after first inference"

    # ── timing assertions ─────────────────────────────────────────────────────
    assert timings[0] < COLD_LIMIT, (
        f"[{entry_point} / {scenario}] "
        f"Cold load took {timings[0]:.3f}s — expected < {COLD_LIMIT}s"
    )
    for i, t in enumerate(timings[1:], start=2):
        assert t < WARM_LIMIT, (
            f"[{entry_point} / {scenario}] "
            f"Run {i} took {t:.3f}s — expected < {WARM_LIMIT}s (model should be cached)"
        )

    return timings


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_direct_default_model(indoor_image_bytes, tmp_path, timing_collector):
    """5 inferences with the default model backend (downloads to vizion3d cache)."""
    _run_group(
        model_backend=DEFAULT_DEPTH_MODEL_URL,
        indoor_image_bytes=indoor_image_bytes,
        run_dir=tmp_path / "direct_default",
        entry_point="Direct",
        scenario="Default model",
        timing_collector=timing_collector,
    )


def test_direct_local_model(
    indoor_image_bytes, local_model_path, tmp_path, timing_collector
):
    """5 inferences with an explicit local .pth path in a tmp directory."""
    _run_group(
        model_backend=local_model_path,
        indoor_image_bytes=indoor_image_bytes,
        run_dir=tmp_path / "direct_local",
        entry_point="Direct",
        scenario="Local model",
        timing_collector=timing_collector,
    )
