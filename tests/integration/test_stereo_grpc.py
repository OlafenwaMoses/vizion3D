"""
Integration tests for the Stereo Depth gRPC entry point.

A live in-process gRPC server is provided by the session fixture. These tests
exercise real S2M2 inference through RunStereoDepth.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("open3d", reason="open3d required - run: uv python pin 3.12 && uv sync")

from vizion3d.proto import lifting_pb2  # noqa: E402
from vizion3d.stereo.defaults import DEFAULT_STEREO_MODEL_URL  # noqa: E402
from vizion3d.stereo.handlers import StereoDepthHandler  # noqa: E402

N_RUNS = 5
COLD_LIMIT = float(os.environ.get("VIZION3D_TEST_STEREO_COLD_LIMIT", "60.0"))
WARM_LIMIT = float(os.environ.get("VIZION3D_TEST_STEREO_WARM_LIMIT", "5.0"))


def _proto_config(config, **overrides):
    values = {
        "focal_length": config.focal_length,
        "baseline": config.baseline,
        "cx": config.cx,
        "cy": config.cy,
        "doffs": config.doffs,
        "z_far": config.z_far,
        "conf_threshold": config.conf_threshold,
        "occ_threshold": config.occ_threshold,
        "scale_factor": config.scale_factor,
    }
    values.update(overrides)
    return lifting_pb2.StereoDepthAdvancedConfig(**values)


def _save_outputs(response, run_dir: Path, run: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(run_dir / f"run_{run:02d}")

    depth_map = [list(row.values) for row in response.depth_map]
    disparity_map = [list(row.values) for row in response.disparity_map]
    Path(prefix + ".depth_map.json").write_text(json.dumps(depth_map))
    Path(prefix + ".disparity_map.json").write_text(json.dumps(disparity_map))

    if response.depth_image:
        Path(prefix + "_depth.png").write_bytes(response.depth_image)

    if response.point_cloud_ply:
        Path(prefix + "_point_cloud.ply").write_bytes(response.point_cloud_ply)


def _run_group(
    model_backend: str,
    stereo_image_pair: tuple[bytes, bytes],
    stereo_advanced_config,
    run_dir: Path,
    scenario: str,
    grpc_client_stub,
    timing_collector,
) -> list[float]:
    StereoDepthHandler._stereo_models.clear()
    left_bytes, right_bytes = stereo_image_pair
    timings: list[float] = []

    for run in range(1, N_RUNS + 1):
        request = lifting_pb2.StereoDepthRequest(
            left_image_bytes=left_bytes,
            right_image_bytes=right_bytes,
            model_backend=model_backend,
            return_depth_image=True,
            return_point_cloud=True,
            advanced_config=_proto_config(stereo_advanced_config),
        )

        t0 = time.perf_counter()
        response = grpc_client_stub.RunStereoDepth(request)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        _save_outputs(response, run_dir, run)
        timing_collector.add("Stereo gRPC", scenario, run, elapsed, str(run_dir))

        assert len(response.depth_map) > 0
        assert len(response.disparity_map) > 0
        assert response.max_depth >= response.min_depth
        assert response.depth_image[:4] == b"\x89PNG"
        assert response.point_cloud_ply.startswith(b"ply\n")

    assert len(StereoDepthHandler._stereo_models) > 0
    assert timings[0] < COLD_LIMIT
    for elapsed in timings[1:]:
        assert elapsed < WARM_LIMIT

    return timings


def test_stereo_grpc_default_model(
    stereo_image_pair, stereo_advanced_config, grpc_client_stub, tmp_path, timing_collector
):
    _run_group(
        model_backend=DEFAULT_STEREO_MODEL_URL,
        stereo_image_pair=stereo_image_pair,
        stereo_advanced_config=stereo_advanced_config,
        run_dir=tmp_path / "stereo_grpc_default",
        scenario="Default model",
        grpc_client_stub=grpc_client_stub,
        timing_collector=timing_collector,
    )


def test_stereo_grpc_local_model(
    stereo_image_pair,
    stereo_advanced_config,
    local_stereo_model_path,
    grpc_client_stub,
    tmp_path,
    timing_collector,
):
    _run_group(
        model_backend=local_stereo_model_path,
        stereo_image_pair=stereo_image_pair,
        stereo_advanced_config=stereo_advanced_config,
        run_dir=tmp_path / "stereo_grpc_local",
        scenario="Local model",
        grpc_client_stub=grpc_client_stub,
        timing_collector=timing_collector,
    )


def test_stereo_grpc_advanced_config_accepted(
    stereo_image_pair, stereo_advanced_config, local_stereo_model_path, grpc_client_stub
):
    StereoDepthHandler._stereo_models.clear()
    left_bytes, right_bytes = stereo_image_pair

    request = lifting_pb2.StereoDepthRequest(
        left_image_bytes=left_bytes,
        right_image_bytes=right_bytes,
        model_backend=local_stereo_model_path,
        return_point_cloud=True,
        advanced_config=_proto_config(stereo_advanced_config, scale_factor=0.5),
    )
    response = grpc_client_stub.RunStereoDepth(request)

    assert len(response.depth_map) > 0
    assert len(response.disparity_map) > 0
    assert response.max_depth >= response.min_depth
    assert response.point_cloud_ply.startswith(b"ply\n")
