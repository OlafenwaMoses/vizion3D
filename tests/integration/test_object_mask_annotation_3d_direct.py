"""
Integration tests — direct Python entry point (ObjectMaskAnnotation3D.run).
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
    ObjectMaskAnnotation3D,
    ObjectMaskAnnotation3DCommand,
)
from vizion3d.annotation.handlers import ObjectMaskAnnotation3DHandler  # noqa: E402
from vizion3d.annotation.models import ObjectMaskAnnotation3DConfig  # noqa: E402
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand  # noqa: E402
from vizion3d.lifting.utils import create_ply_binary  # noqa: E402

N_RUNS = 5
COLD_LIMIT = float(os.environ.get("VIZION3D_TEST_ANNOTATION_COLD_LIMIT", "30.0"))
WARM_LIMIT = float(os.environ.get("VIZION3D_TEST_ANNOTATION_WARM_LIMIT", "2.0"))


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
                "confidence": a.confidence,
                "bbox_2d": a.bbox_2d,
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


def _run_group(
    model_backend,
    indoor_image_bytes,
    indoor_point_cloud,
    run_dir,
    entry_point,
    scenario,
    timing_collector,
):
    ObjectMaskAnnotation3DHandler._annotation_models.clear()
    timings = []
    for run in range(1, N_RUNS + 1):
        t0 = time.perf_counter()
        result = ObjectMaskAnnotation3D().run(
            ObjectMaskAnnotation3DCommand(
                point_cloud=indoor_point_cloud,
                image_input=indoor_image_bytes,
                model_backend=model_backend,
                return_annotated_cloud=True,
            )
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        _save_outputs(result, run_dir, run)
        timing_collector.add(entry_point, scenario, run, elapsed, str(run_dir))

        assert isinstance(result.annotations, list)
        assert isinstance(result.backend_used, str) and result.backend_used
        for ann in result.annotations:
            assert isinstance(ann.label, str) and ann.label
            assert 0.0 <= ann.confidence <= 1.0
            assert len(ann.bbox_2d) == 4
            assert ann.mask_2d.dtype == bool
            assert len(ann.point_indices) == len(ann.point_coords)
            assert all(len(c) == 3 for c in ann.point_coords)
        if result.annotated_cloud is not None:
            n_in = len(np.asarray(indoor_point_cloud.points))
            assert len(np.asarray(result.annotated_cloud.points)) == n_in

    assert len(ObjectMaskAnnotation3DHandler._annotation_models) > 0
    assert timings[0] < COLD_LIMIT
    for i, t in enumerate(timings[1:], start=2):
        assert t < WARM_LIMIT, f"[{entry_point}/{scenario}] Run {i}: {t:.3f}s > {WARM_LIMIT}s"
    return timings


def test_direct_default_model(
    indoor_image_bytes, indoor_point_cloud, local_annotation_model_path, tmp_path, timing_collector
):
    _run_group(
        model_backend=local_annotation_model_path,
        indoor_image_bytes=indoor_image_bytes,
        indoor_point_cloud=indoor_point_cloud,
        run_dir=tmp_path / "direct_annotation_default",
        entry_point="Direct",
        scenario="Default model",
        timing_collector=timing_collector,
    )


def test_direct_custom_intrinsics(
    indoor_image_bytes, indoor_point_cloud, local_annotation_model_path
):
    ObjectMaskAnnotation3DHandler._annotation_models.clear()
    result = ObjectMaskAnnotation3D().run(
        ObjectMaskAnnotation3DCommand(
            point_cloud=indoor_point_cloud,
            image_input=indoor_image_bytes,
            model_backend=local_annotation_model_path,
            advanced_config=ObjectMaskAnnotation3DConfig(fx=615.0, fy=615.0, cx=320.0, cy=240.0),
        )
    )
    assert isinstance(result.annotations, list)
    assert result.backend_used == local_annotation_model_path


def test_direct_return_object_clouds(
    indoor_image_bytes, indoor_point_cloud, local_annotation_model_path
):
    ObjectMaskAnnotation3DHandler._annotation_models.clear()
    result = ObjectMaskAnnotation3D().run(
        ObjectMaskAnnotation3DCommand(
            point_cloud=indoor_point_cloud,
            image_input=indoor_image_bytes,
            model_backend=local_annotation_model_path,
            return_object_clouds=True,
        )
    )
    for ann in result.annotations:
        if ann.point_indices:
            assert ann.object_cloud is not None
            assert isinstance(ann.object_cloud, o3d.geometry.PointCloud)
            assert len(np.asarray(ann.object_cloud.points)) == len(ann.point_indices)


def test_direct_no_image_uses_front_view(indoor_point_cloud, local_annotation_model_path):
    """image_input=None: handler synthesises front view from cloud and runs normally."""
    ObjectMaskAnnotation3DHandler._annotation_models.clear()
    result = ObjectMaskAnnotation3D().run(
        ObjectMaskAnnotation3DCommand(
            point_cloud=indoor_point_cloud,
            model_backend=local_annotation_model_path,
        )
    )
    assert isinstance(result.annotations, list)
    assert result.backend_used == local_annotation_model_path


def test_direct_image_file_path(indoor_point_cloud, local_annotation_model_path, tmp_path):
    from PIL import Image as PILImage

    img_path = tmp_path / "scene.png"
    PILImage.new("RGB", (64, 48), "white").save(img_path)

    ObjectMaskAnnotation3DHandler._annotation_models.clear()
    result = ObjectMaskAnnotation3D().run(
        ObjectMaskAnnotation3DCommand(
            point_cloud=indoor_point_cloud,
            image_input=str(img_path),
            model_backend=local_annotation_model_path,
        )
    )
    assert isinstance(result.annotations, list)
