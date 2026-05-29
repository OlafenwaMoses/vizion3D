"""
Integration tests — gRPC entry point (RunSceneMaskAnnotation3D).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytest.importorskip("open3d", reason="open3d required")

from vizion3d.annotation.scene_handlers import SceneMaskAnnotation3DHandler  # noqa: E402
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand  # noqa: E402
from vizion3d.lifting.utils import create_ply_binary  # noqa: E402
from vizion3d.proto import lifting_pb2  # noqa: E402

COLD_LIMIT = float(os.environ.get("VIZION3D_TEST_SCENE_COLD_LIMIT", "60.0"))
WARM_LIMIT = float(os.environ.get("VIZION3D_TEST_SCENE_WARM_LIMIT", "8.0"))
N_RUNS = 5


@pytest.fixture(scope="session")
def indoor_point_cloud_ply_bytes(indoor_image_bytes, local_model_path) -> bytes:
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


def test_grpc_basic_scene_annotation(
    indoor_image_bytes,
    indoor_point_cloud_ply_bytes,
    local_scene_model_path,
    grpc_client_stub,
    tmp_path,
    timing_collector,
):
    SceneMaskAnnotation3DHandler._scene_models.clear()
    timings = []
    for run in range(1, N_RUNS + 1):
        request = lifting_pb2.SceneMaskAnnotation3DRequest(
            image_bytes=indoor_image_bytes,
            point_cloud_ply=indoor_point_cloud_ply_bytes,
            model_backend=local_scene_model_path,
            return_annotated_cloud=True,
        )
        t0 = time.perf_counter()
        response = grpc_client_stub.RunSceneMaskAnnotation3D(request)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        timing_collector.add("gRPC", "Scene default", run, elapsed, str(tmp_path))

        assert isinstance(response.backend_used, str) and response.backend_used
        assert len(response.annotations) >= 1
        for item in response.annotations:
            assert isinstance(item.label, str) and item.label
            assert 0 <= item.class_id < 150
            assert item.pixel_count > 0
            assert item.mask_image[:4] == b"\x89PNG"
        if response.annotated_cloud_ply:
            assert response.annotated_cloud_ply.startswith(b"ply\n")

    assert len(SceneMaskAnnotation3DHandler._scene_models) > 0
    assert timings[0] < COLD_LIMIT
    for i, t in enumerate(timings[1:], start=2):
        assert t < WARM_LIMIT, f"[gRPC/Scene] Run {i}: {t:.3f}s > {WARM_LIMIT}s"


def test_grpc_region_clouds_returned(
    indoor_image_bytes,
    indoor_point_cloud_ply_bytes,
    local_scene_model_path,
    grpc_client_stub,
):
    SceneMaskAnnotation3DHandler._scene_models.clear()
    request = lifting_pb2.SceneMaskAnnotation3DRequest(
        image_bytes=indoor_image_bytes,
        point_cloud_ply=indoor_point_cloud_ply_bytes,
        model_backend=local_scene_model_path,
        return_region_clouds=True,
    )
    response = grpc_client_stub.RunSceneMaskAnnotation3D(request)
    for item in response.annotations:
        if item.point_indices:
            assert item.region_cloud_ply and item.region_cloud_ply.startswith(b"ply\n")


def test_grpc_custom_config(
    indoor_image_bytes,
    indoor_point_cloud_ply_bytes,
    local_scene_model_path,
    grpc_client_stub,
):
    SceneMaskAnnotation3DHandler._scene_models.clear()
    proto_cfg = lifting_pb2.SceneMaskAnnotation3DConfig(
        fx=615.0, fy=615.0, cx=320.0, cy=240.0, inference_size=384
    )
    request = lifting_pb2.SceneMaskAnnotation3DRequest(
        image_bytes=indoor_image_bytes,
        point_cloud_ply=indoor_point_cloud_ply_bytes,
        model_backend=local_scene_model_path,
        advanced_config=proto_cfg,
    )
    response = grpc_client_stub.RunSceneMaskAnnotation3D(request)
    assert isinstance(response.backend_used, str)


def test_grpc_no_image_uses_front_view(
    indoor_point_cloud_ply_bytes, local_scene_model_path, grpc_client_stub
):
    """Empty image_bytes → handler synthesises front view from point cloud."""
    SceneMaskAnnotation3DHandler._scene_models.clear()
    request = lifting_pb2.SceneMaskAnnotation3DRequest(
        point_cloud_ply=indoor_point_cloud_ply_bytes,
        model_backend=local_scene_model_path,
    )
    response = grpc_client_stub.RunSceneMaskAnnotation3D(request)
    assert isinstance(response.backend_used, str)
