"""
Integration tests — direct Python entry point (SceneMaskAnnotation3D.run).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("open3d", reason="open3d required")

import open3d as o3d  # noqa: E402

from vizion3d.annotation import (  # noqa: E402
    SceneMaskAnnotation3D,
    SceneMaskAnnotation3DCommand,
)
from vizion3d.annotation.models import SceneMaskAnnotation3DConfig  # noqa: E402
from vizion3d.annotation.scene_handlers import SceneMaskAnnotation3DHandler  # noqa: E402
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand  # noqa: E402
from vizion3d.lifting.utils import create_ply_binary  # noqa: E402

N_RUNS = 5
COLD_LIMIT = float(os.environ.get("VIZION3D_TEST_SCENE_COLD_LIMIT", "60.0"))
WARM_LIMIT = float(os.environ.get("VIZION3D_TEST_SCENE_WARM_LIMIT", "8.0"))


@pytest.fixture(scope="session")
def indoor_point_cloud(indoor_image_bytes, local_model_path):
    result = DepthEstimation().run(
        DepthEstimationCommand(
            image_input=indoor_image_bytes,
            model_backend=local_model_path,
            return_point_cloud=True,
            return_depth_image=False,
        )
    )
    assert result.point_cloud is not None
    return result.point_cloud


def _save_outputs(result, run_dir: Path, run: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_dir / f"run_{run:02d}"
    summary = {
        "num_annotations": len(result.annotations),
        "backend_used": result.backend_used,
        "annotations": [
            {
                "label": a.label,
                "class_id": a.class_id,
                "pixel_count": a.pixel_count,
                "num_points": len(a.point_indices),
            }
            for a in result.annotations
        ],
    }
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    if result.annotated_cloud is not None and result.annotated_cloud.has_points():
        pts = np.asarray(result.annotated_cloud.points).astype(np.float32)
        cols = (np.asarray(result.annotated_cloud.colors) * 255).astype(np.uint8)
        prefix.with_suffix(".ply").write_bytes(create_ply_binary(pts, cols))


def test_direct_default_model(
    indoor_image_bytes, indoor_point_cloud, local_scene_model_path, tmp_path, timing_collector
):
    SceneMaskAnnotation3DHandler._scene_models.clear()
    run_dir = tmp_path / "direct_scene_default"
    timings = []
    for run in range(1, N_RUNS + 1):
        t0 = time.perf_counter()
        result = SceneMaskAnnotation3D().run(
            SceneMaskAnnotation3DCommand(
                point_cloud=indoor_point_cloud,
                image_input=indoor_image_bytes,
                model_backend=local_scene_model_path,
                return_annotated_cloud=True,
            )
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        _save_outputs(result, run_dir, run)
        timing_collector.add("Direct", "Scene default", run, elapsed, str(run_dir))

        assert isinstance(result.annotations, list)
        assert len(result.annotations) >= 1
        assert isinstance(result.backend_used, str) and result.backend_used
        for ann in result.annotations:
            assert isinstance(ann.label, str) and ann.label
            assert 0 <= ann.class_id < 150
            assert ann.mask_2d.dtype == bool
            assert ann.pixel_count == int(ann.mask_2d.sum())
            assert len(ann.point_indices) == len(ann.point_coords)
        # annotations sorted by descending pixel count
        counts = [a.pixel_count for a in result.annotations]
        assert counts == sorted(counts, reverse=True)
        if result.annotated_cloud is not None:
            n_in = len(np.asarray(indoor_point_cloud.points))
            assert len(np.asarray(result.annotated_cloud.points)) == n_in

    assert len(SceneMaskAnnotation3DHandler._scene_models) > 0
    assert timings[0] < COLD_LIMIT
    for i, t in enumerate(timings[1:], start=2):
        assert t < WARM_LIMIT, f"[Direct/Scene] Run {i}: {t:.3f}s > {WARM_LIMIT}s"


def test_direct_region_clouds(indoor_image_bytes, indoor_point_cloud, local_scene_model_path):
    SceneMaskAnnotation3DHandler._scene_models.clear()
    result = SceneMaskAnnotation3D().run(
        SceneMaskAnnotation3DCommand(
            point_cloud=indoor_point_cloud,
            image_input=indoor_image_bytes,
            model_backend=local_scene_model_path,
            return_region_clouds=True,
        )
    )
    for ann in result.annotations:
        if ann.point_indices:
            assert isinstance(ann.region_cloud, o3d.geometry.PointCloud)
            assert len(np.asarray(ann.region_cloud.points)) == len(ann.point_indices)


def test_direct_custom_inference_size(
    indoor_image_bytes, indoor_point_cloud, local_scene_model_path
):
    SceneMaskAnnotation3DHandler._scene_models.clear()
    result = SceneMaskAnnotation3D().run(
        SceneMaskAnnotation3DCommand(
            point_cloud=indoor_point_cloud,
            image_input=indoor_image_bytes,
            model_backend=local_scene_model_path,
            advanced_config=SceneMaskAnnotation3DConfig(inference_size=384),
        )
    )
    assert isinstance(result.annotations, list)
    assert result.backend_used == local_scene_model_path


def test_direct_no_image_uses_front_view(indoor_point_cloud, local_scene_model_path):
    """image_input=None: handler synthesises front view from cloud and runs."""
    SceneMaskAnnotation3DHandler._scene_models.clear()
    result = SceneMaskAnnotation3D().run(
        SceneMaskAnnotation3DCommand(
            point_cloud=indoor_point_cloud,
            model_backend=local_scene_model_path,
        )
    )
    assert isinstance(result.annotations, list)
    assert result.backend_used == local_scene_model_path
