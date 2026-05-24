"""
Integration tests — ScaleObservation direct, REST, and gRPC entry points.

These tests use the existing integration asset ``tests/assets/indoor_scene.jpg``.
The point cloud is produced by the DepthEstimation task and annotations are
produced by ObjectMaskAnnotation3D, then passed into ScaleObservation as
``annotation_result.annotations``.
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

from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand  # noqa: E402
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand  # noqa: E402
from vizion3d.lifting.utils import create_ply_binary  # noqa: E402
from vizion3d.observation import (  # noqa: E402
    ScaleObservation,
    ScaleObservationAdvancedConfig,
    ScaleObservationCommand,
)
from vizion3d.proto import lifting_pb2  # noqa: E402
from vizion3d.server.grpc.server import _mask_to_png_bytes  # noqa: E402
from vizion3d.server.rest.app import app  # noqa: E402

N_RUNS = 5
SCALE_LIMIT = float(os.environ.get("VIZION3D_TEST_SCALE_OBSERVATION_LIMIT", "3.0"))
MAX_WIRE_POINTS_PER_ANNOTATION = 1000
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FX = 544.0
FY = 544.0
CX = 320.0
CY = 240.0

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="session")
def scale_observation_point_cloud(indoor_image_bytes, local_model_path):
    result = DepthEstimation().run(
        DepthEstimationCommand(
            image_input=indoor_image_bytes,
            model_backend=local_model_path,
            return_depth_image=False,
            return_point_cloud=True,
        )
    )
    assert result.point_cloud is not None
    assert result.point_cloud.has_points()
    return result.point_cloud


@pytest.fixture(scope="session")
def scale_observation_annotations(
    indoor_image_bytes,
    scale_observation_point_cloud,
    local_annotation_model_path,
):
    result = ObjectMaskAnnotation3D().run(
        ObjectMaskAnnotation3DCommand(
            point_cloud=scale_observation_point_cloud,
            image_input=indoor_image_bytes,
            model_backend=local_annotation_model_path,
        )
    )
    assert isinstance(result.annotations, list)
    return result.annotations


@pytest.fixture(scope="session")
def scale_observation_point_cloud_ply(scale_observation_point_cloud) -> bytes:
    pts = np.asarray(scale_observation_point_cloud.points).astype(np.float32)
    cols = (np.asarray(scale_observation_point_cloud.colors) * 255).astype(np.uint8)
    return create_ply_binary(pts, cols)


@pytest.fixture(scope="session")
def scale_observation_annotations_json(scale_observation_annotations) -> str:
    return json.dumps(_wire_annotations(scale_observation_annotations))


@pytest.fixture(scope="session")
def scale_observation_wire_annotations(scale_observation_annotations) -> list[dict]:
    return _wire_annotations(scale_observation_annotations)


def _wire_annotations(annotations) -> list[dict]:
    payload = []
    for ann in annotations:
        coords = ann.point_coords
        indices = ann.point_indices
        if len(coords) > MAX_WIRE_POINTS_PER_ANNOTATION:
            sample_idx = np.linspace(
                0,
                len(coords) - 1,
                MAX_WIRE_POINTS_PER_ANNOTATION,
                dtype=np.int64,
            )
            coords = [coords[int(i)] for i in sample_idx]
            indices = [indices[int(i)] for i in sample_idx]
        payload.append(
            {
                "label": ann.label,
                "class_id": ann.class_id,
                "confidence": ann.confidence,
                "bbox_2d": ann.bbox_2d,
                "mask_2d": ann.mask_2d.tolist(),
                "point_indices": indices,
                "point_coords": coords,
            }
        )
    return payload


def _advanced_config() -> ScaleObservationAdvancedConfig:
    return ScaleObservationAdvancedConfig(
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )


def _save_direct_outputs(result, run_dir: Path, run: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_dir / f"run_{run:02d}"
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "scale_factor": result.scale_factor,
                "scale_confidence": result.scale_confidence,
                "algorithm_version": result.algorithm_version,
                "accepted_candidates": result.accepted_candidates,
                "rejected_candidates": result.rejected_candidates,
                "scale_report": result.scale_report,
            },
            indent=2,
        )
    )
    if result.scaled_point_cloud is not None:
        pts = np.asarray(result.scaled_point_cloud.points).astype(np.float32)
        cols = (np.asarray(result.scaled_point_cloud.colors) * 255).astype(np.uint8)
        prefix.with_suffix(".ply").write_bytes(create_ply_binary(pts, cols))
    if result.scaled_depth_image is not None:
        np.save(str(prefix) + "_scaled_depth.npy", np.asarray(result.scaled_depth_image))


def _assert_scale_output(result) -> None:
    assert result.algorithm_version == "v4_iter_402_lower_quantile_mean_blend"
    assert result.scale_factor > 0
    assert 0.0 <= result.scale_confidence <= 1.0
    assert result.accepted_candidates + result.rejected_candidates == len(result.candidates)
    assert isinstance(result.scale_confidence_reason, str)
    assert "scale_factor" in result.scale_report
    assert "generated_bounds" in result.scale_report
    assert "scaled_bounds" in result.scale_report


def test_scale_observation_direct_outputs_params_advanced_config_and_timing(
    scale_observation_point_cloud,
    scale_observation_annotations,
    tmp_path,
    timing_collector,
):
    timings = []
    run_dir = tmp_path / "scale_observation_direct"
    for run in range(1, N_RUNS + 1):
        t0 = time.perf_counter()
        result = ScaleObservation().run(
            ScaleObservationCommand(
                point_cloud=scale_observation_point_cloud,
                annotations=scale_observation_annotations,
                return_scaled_point_cloud=True,
                return_scaled_depth=True,
                return_report=True,
                advanced_config=_advanced_config(),
            )
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        timing_collector.add("Scale Direct", "Indoor asset", run, elapsed, str(run_dir))
        _save_direct_outputs(result, run_dir, run)

        _assert_scale_output(result)
        assert result.scaled_point_cloud is not None
        assert result.scaled_point_cloud.has_points()
        assert result.scaled_depth_image is not None
        assert np.asarray(result.scaled_depth_image).shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
        assert result.scaled_depth_metadata is not None

    for run, elapsed in enumerate(timings, start=1):
        assert elapsed < SCALE_LIMIT, f"[Scale Direct] Run {run}: {elapsed:.3f}s > {SCALE_LIMIT}s"


def test_scale_observation_rest_outputs_params_advanced_config_and_timing(
    scale_observation_point_cloud_ply,
    scale_observation_annotations_json,
    tmp_path,
    timing_collector,
):
    timings = []
    run_dir = tmp_path / "scale_observation_rest"
    run_dir.mkdir(parents=True, exist_ok=True)
    for run in range(1, N_RUNS + 1):
        t0 = time.perf_counter()
        response = client.post(
            "/observation/scale-observation",
            files={
                "point_cloud": (
                    "cloud.ply",
                    scale_observation_point_cloud_ply,
                    "application/octet-stream",
                ),
                "annotations_file": (
                    "annotations.json",
                    scale_observation_annotations_json.encode("utf-8"),
                    "application/json",
                ),
            },
            data={
                "return_scaled_point_cloud": "true",
                "return_scaled_depth": "true",
                "return_report": "true",
                "image_width": str(IMAGE_WIDTH),
                "image_height": str(IMAGE_HEIGHT),
                "fx": str(FX),
                "fy": str(FY),
                "cx": str(CX),
                "cy": str(CY),
            },
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        timing_collector.add("Scale REST", "Indoor asset", run, elapsed, str(run_dir))

        assert response.status_code == 200, response.text[:300]
        data = response.json()
        Path(run_dir / f"run_{run:02d}.json").write_text(json.dumps(data, indent=2))

        assert data["algorithm_version"] == "v4_iter_402_lower_quantile_mean_blend"
        assert data["scale_factor"] > 0
        assert 0.0 <= data["scale_confidence"] <= 1.0
        assert data["accepted_candidates"] + data["rejected_candidates"] == len(data["candidates"])
        assert data["scaled_point_cloud_ply"] is not None
        assert base64.b64decode(data["scaled_point_cloud_ply"]).startswith(b"ply\n")
        assert data["scaled_depth_png"] is not None
        assert base64.b64decode(data["scaled_depth_png"])[:4] == b"\x89PNG"
        assert data["scaled_depth_metadata"]["units"] == "metres"
        assert "scaled_bounds" in data["scale_report"]

    for run, elapsed in enumerate(timings, start=1):
        assert elapsed < SCALE_LIMIT, f"[Scale REST] Run {run}: {elapsed:.3f}s > {SCALE_LIMIT}s"


def test_scale_observation_grpc_outputs_params_advanced_config_and_timing(
    scale_observation_point_cloud_ply,
    scale_observation_wire_annotations,
    grpc_client_stub,
    tmp_path,
    timing_collector,
):
    timings = []
    run_dir = tmp_path / "scale_observation_grpc"
    run_dir.mkdir(parents=True, exist_ok=True)
    for run in range(1, N_RUNS + 1):
        request = lifting_pb2.ScaleObservationRequest(
            point_cloud_ply=scale_observation_point_cloud_ply,
            return_scaled_point_cloud=True,
            return_scaled_depth=True,
            return_report=True,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            fx=FX,
            fy=FY,
            cx=CX,
            cy=CY,
        )
        for ann in scale_observation_wire_annotations:
            item = request.annotations.add(
                label=ann["label"],
                class_id=ann["class_id"],
                confidence=ann["confidence"],
                bbox_2d=ann["bbox_2d"],
            )
            item.mask_image = _mask_to_png_bytes(np.asarray(ann["mask_2d"], dtype=bool))
            for coord in ann["point_coords"]:
                item.point_coords.append(lifting_pb2.FloatRow(values=coord))

        t0 = time.perf_counter()
        response = grpc_client_stub.RunScaleObservation(request)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        timing_collector.add("Scale gRPC", "Indoor asset", run, elapsed, str(run_dir))

        Path(run_dir / f"run_{run:02d}.json").write_text(
            json.dumps(
                {
                    "scale_factor": response.scale_factor,
                    "scale_confidence": response.scale_confidence,
                    "algorithm_version": response.algorithm_version,
                    "accepted_candidates": response.accepted_candidates,
                    "rejected_candidates": response.rejected_candidates,
                },
                indent=2,
            )
        )

        assert response.algorithm_version == "v4_iter_402_lower_quantile_mean_blend"
        assert response.scale_factor > 0
        assert 0.0 <= response.scale_confidence <= 1.0
        assert response.accepted_candidates + response.rejected_candidates == len(
            response.candidates
        )
        assert response.scaled_point_cloud_ply.startswith(b"ply\n")
        assert response.scaled_depth_png[:4] == b"\x89PNG"

    for run, elapsed in enumerate(timings, start=1):
        assert elapsed < SCALE_LIMIT, f"[Scale gRPC] Run {run}: {elapsed:.3f}s > {SCALE_LIMIT}s"
