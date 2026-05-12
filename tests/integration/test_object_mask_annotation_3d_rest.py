"""
Integration tests — REST endpoint POST /annotation/object-mask-annotation-3d.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("open3d", reason="open3d required")

from vizion3d.annotation.handlers import ObjectMaskAnnotation3DHandler  # noqa: E402
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand  # noqa: E402
from vizion3d.lifting.utils import create_ply_binary  # noqa: E402
from vizion3d.server.rest.app import app  # noqa: E402

N_RUNS = 5
COLD_LIMIT = float(os.environ.get("VIZION3D_TEST_ANNOTATION_COLD_LIMIT", "30.0"))
WARM_LIMIT = float(os.environ.get("VIZION3D_TEST_ANNOTATION_WARM_LIMIT", "2.0"))

client = TestClient(app, raise_server_exceptions=True)

ENDPOINT = "/annotation/object-mask-annotation-3d"


@pytest.fixture(scope="session")
def indoor_point_cloud_ply(indoor_image_bytes, local_model_path) -> bytes:
    result = DepthEstimation().run(
        DepthEstimationCommand(
            image_input=indoor_image_bytes,
            model_backend=local_model_path,
            return_point_cloud=True,
            return_depth_image=False,
        )
    )
    assert result.point_cloud is not None
    pts = np.asarray(result.point_cloud.points).astype(np.float32)
    cols = (np.asarray(result.point_cloud.colors) * 255).astype(np.uint8)
    return create_ply_binary(pts, cols)


def _save_outputs(data: dict, run_dir: Path, run: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(run_dir / f"run_{run:02d}")
    Path(prefix + ".json").write_text(json.dumps({
        "num_annotations": len(data["annotations"]),
        "backend_used": data["backend_used"],
        "labels": [a["label"] for a in data["annotations"]],
    }, indent=2))
    if data.get("annotated_cloud_ply"):
        Path(prefix + "_annotated.ply").write_bytes(
            base64.b64decode(data["annotated_cloud_ply"])
        )


def _run_group(model_backend, indoor_image_bytes, indoor_point_cloud_ply,
               run_dir, scenario, timing_collector):
    ObjectMaskAnnotation3DHandler._annotation_models.clear()
    timings = []
    for run in range(1, N_RUNS + 1):
        t0 = time.perf_counter()
        response = client.post(
            ENDPOINT,
            files={
                "image": ("scene.jpg", indoor_image_bytes, "image/jpeg"),
                "point_cloud_ply": ("cloud.ply", indoor_point_cloud_ply, "application/octet-stream"),
            },
            data={"model_backend": model_backend, "return_annotated_cloud": "true"},
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        assert response.status_code == 200, response.text[:300]
        data = response.json()
        _save_outputs(data, run_dir, run)
        timing_collector.add("REST", scenario, run, elapsed, str(run_dir))

        assert "annotations" in data and "backend_used" in data
        for ann in data["annotations"]:
            assert isinstance(ann["label"], str)
            assert 0.0 <= ann["confidence"] <= 1.0
            assert len(ann["bbox_2d"]) == 4
            mask_bytes = base64.b64decode(ann["mask_image"])
            assert mask_bytes[:4] == b"\x89PNG"
            assert isinstance(ann["point_indices"], list)
        if data.get("annotated_cloud_ply"):
            assert base64.b64decode(data["annotated_cloud_ply"]).startswith(b"ply\n")

    assert len(ObjectMaskAnnotation3DHandler._annotation_models) > 0
    assert timings[0] < COLD_LIMIT
    for i, t in enumerate(timings[1:], start=2):
        assert t < WARM_LIMIT, f"[REST/{scenario}] Run {i}: {t:.3f}s > {WARM_LIMIT}s"
    return timings


def test_rest_default_model(
    indoor_image_bytes, indoor_point_cloud_ply, local_annotation_model_path,
    tmp_path, timing_collector,
):
    _run_group(
        model_backend=local_annotation_model_path,
        indoor_image_bytes=indoor_image_bytes,
        indoor_point_cloud_ply=indoor_point_cloud_ply,
        run_dir=tmp_path / "rest_annotation_default",
        scenario="Default model",
        timing_collector=timing_collector,
    )


def test_rest_custom_intrinsics(
    indoor_image_bytes, indoor_point_cloud_ply, local_annotation_model_path
):
    ObjectMaskAnnotation3DHandler._annotation_models.clear()
    resp = client.post(
        ENDPOINT,
        files={
            "image": ("scene.jpg", indoor_image_bytes, "image/jpeg"),
            "point_cloud_ply": ("cloud.ply", indoor_point_cloud_ply, "application/octet-stream"),
        },
        data={
            "model_backend": local_annotation_model_path,
            "fx": "615.0", "fy": "615.0", "cx": "320.0", "cy": "240.0",
        },
    )
    assert resp.status_code == 200
    assert "annotations" in resp.json()


def test_rest_object_clouds_returned(
    indoor_image_bytes, indoor_point_cloud_ply, local_annotation_model_path
):
    ObjectMaskAnnotation3DHandler._annotation_models.clear()
    resp = client.post(
        ENDPOINT,
        files={
            "image": ("scene.jpg", indoor_image_bytes, "image/jpeg"),
            "point_cloud_ply": ("cloud.ply", indoor_point_cloud_ply, "application/octet-stream"),
        },
        data={"model_backend": local_annotation_model_path, "return_object_clouds": "true"},
    )
    assert resp.status_code == 200
    for ann in resp.json()["annotations"]:
        if ann["point_indices"]:
            assert ann["object_cloud_ply"] is not None
            assert base64.b64decode(ann["object_cloud_ply"]).startswith(b"ply\n")


def test_rest_no_image_uses_front_view(indoor_point_cloud_ply, local_annotation_model_path):
    """Omitting the image file is accepted — cloud front-view is synthesised."""
    ObjectMaskAnnotation3DHandler._annotation_models.clear()
    resp = client.post(
        ENDPOINT,
        files={
            "point_cloud_ply": ("cloud.ply", indoor_point_cloud_ply, "application/octet-stream"),
        },
        data={"model_backend": local_annotation_model_path},
    )
    assert resp.status_code == 200
    assert "annotations" in resp.json()
