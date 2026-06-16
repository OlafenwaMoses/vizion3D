"""
Integration tests — gRPC entry points for 3D reconstruction.

These tests use the live in-process gRPC server fixture with the saved 480px
reconstruction image and low-cost mesh settings.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("open3d", reason="open3d required")
pytest.importorskip("trimesh", reason="trimesh required")
pytest.importorskip("rembg", reason="rembg required")
pytest.importorskip("mcubes", reason="PyMCubes required")
pytest.importorskip("omegaconf", reason="omegaconf required")
pytest.importorskip("einops", reason="einops required")

from vizion3d.proto import lifting_pb2  # noqa: E402
from vizion3d.reconstruction.handlers import Object3DReconstructionHandler  # noqa: E402

OBJECT_LIMIT = float(os.environ.get("VIZION3D_TEST_RECON_GRPC_OBJECT_LIMIT", "260.0"))
SCENE_LIMIT = float(os.environ.get("VIZION3D_TEST_RECON_GRPC_SCENE_LIMIT", "460.0"))
TEST_IMAGE_MAX_DIMENSION = 1080
RECONSTRUCTION_IMAGE = "reconstruction_scene_1080.jpg"


@pytest.fixture(scope="session")
def reconstruction_model_bundle() -> str:
    bundle = Path(__file__).resolve().parents[2] / "scene-components-3d-models.zip"
    if not bundle.is_file():
        pytest.skip(f"Reconstruction model bundle not found: {bundle}")
    return str(bundle)


@pytest.fixture(scope="session", autouse=True)
def triposr_source_available():
    source = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "3D_Object-Reconstruction"
        / "TripoSR"
    )
    if not (source / "tsr" / "system.py").is_file():
        pytest.skip(f"TripoSR source not found: {source}")
    os.environ["VIZION3D_TRIPOSR_SOURCE"] = str(source)


@pytest.fixture(scope="session")
def reconstruction_image_bytes() -> bytes:
    path = Path(__file__).parent.parent / "assets" / RECONSTRUCTION_IMAGE
    if not path.is_file():
        pytest.skip(f"Reconstruction integration image not found: {path}")
    image = Image.open(path)
    assert max(image.size) <= TEST_IMAGE_MAX_DIMENSION
    return path.read_bytes()


def _object_config(point_count: int = 1024) -> lifting_pb2.Object3DReconstructionConfig:
    return lifting_pb2.Object3DReconstructionConfig(
        max_input_dimension=512,
        marching_cubes_resolution=64,
        point_count=point_count,
        smoothing_iterations=0,
        device=os.environ.get("VIZION3D_TEST_RECON_DEVICE", "cpu"),
    )


def _save_object_response(response, run_dir: Path, stem: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    assert response.mesh_ply.startswith(b"ply\n")
    assert response.point_cloud_ply.startswith(b"ply\n")
    (run_dir / f"{stem}_mesh.ply").write_bytes(response.mesh_ply)
    (run_dir / f"{stem}_point_cloud.ply").write_bytes(response.point_cloud_ply)


def _assert_object_response(response) -> None:
    assert response.vertex_count > 0
    assert response.face_count > 0
    assert response.point_count > 0
    assert response.mesh_ply.startswith(b"ply\n")
    assert response.point_cloud_ply.startswith(b"ply\n")


def _poll_grpc_result(grpc_client_stub, method_name: str, job_id: str, timeout: float):
    deadline = time.monotonic() + timeout
    method = getattr(grpc_client_stub, method_name)
    request = lifting_pb2.ReconstructionJobResultRequest(job_id=job_id)
    pending = {
        lifting_pb2.RECONSTRUCTION_JOB_STATUS_QUEUED,
        lifting_pb2.RECONSTRUCTION_JOB_STATUS_RUNNING,
    }
    while time.monotonic() < deadline:
        response = method(request)
        if response.status == lifting_pb2.RECONSTRUCTION_JOB_STATUS_SUCCEEDED:
            assert response.HasField("result")
            return response.result
        assert response.status in pending, response
        time.sleep(1.0)
    pytest.fail(f"Timed out waiting for reconstruction job {job_id}")


def test_grpc_object_3d_reconstruction_runs_real_image(
    reconstruction_image_bytes,
    reconstruction_model_bundle,
    grpc_client_stub,
    tmp_path,
    timing_collector,
):
    Object3DReconstructionHandler._models.clear()
    request = lifting_pb2.Object3DReconstructionRequest(
        image_bytes=reconstruction_image_bytes,
        model_bundle=reconstruction_model_bundle,
        advanced_config=_object_config(),
    )

    t0 = time.perf_counter()
    submission = grpc_client_stub.RunObject3DReconstruction(request)
    response = _poll_grpc_result(
        grpc_client_stub,
        "GetObject3DReconstructionResult",
        submission.job_id,
        OBJECT_LIMIT,
    )
    elapsed = time.perf_counter() - t0

    _assert_object_response(response)
    assert response.backend_used
    run_dir = tmp_path / "grpc_object_3d_reconstruction"
    _save_object_response(response, run_dir, "object")
    timing_collector.add(
        "gRPC",
        "TripoSR object",
        1,
        elapsed,
        str(run_dir),
        task="Object 3D Reconstruction",
        model="TripoSR",
        device=os.environ.get("VIZION3D_TEST_RECON_DEVICE", "cpu"),
    )
    assert elapsed < OBJECT_LIMIT


def test_grpc_scene_components_3d_reconstruction_runs_real_image(
    reconstruction_image_bytes,
    reconstruction_model_bundle,
    local_model_path,
    local_annotation_model_path,
    grpc_client_stub,
    tmp_path,
    timing_collector,
):
    Object3DReconstructionHandler._models.clear()
    request = lifting_pb2.SceneComponents3DReconstructionRequest(
        image_bytes=reconstruction_image_bytes,
        model_bundle=reconstruction_model_bundle,
        depth_model_backend=local_model_path,
        annotation_model_backend=local_annotation_model_path,
        advanced_config=lifting_pb2.SceneComponents3DReconstructionConfig(
            max_input_dimension=640,
            max_objects=1,
            confidence_threshold=0.01,
            padding_ratio=0.1,
            object_config=_object_config(point_count=1024),
        ),
    )

    t0 = time.perf_counter()
    submission = grpc_client_stub.RunSceneComponents3DReconstruction(request)
    response = _poll_grpc_result(
        grpc_client_stub,
        "GetSceneComponents3DReconstructionResult",
        submission.job_id,
        SCENE_LIMIT,
    )
    elapsed = time.perf_counter() - t0

    assert response.source_image_size
    assert max(response.source_image_size) <= TEST_IMAGE_MAX_DIMENSION
    assert max(response.analysis_image_size) <= 640
    assert response.depth_backend_used
    assert response.annotation_backend_used
    assert response.reconstruction_backend_used
    assert len(response.components) >= 1

    run_dir = tmp_path / "grpc_scene_components_3d_reconstruction"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_image_size": list(response.source_image_size),
        "analysis_image_size": list(response.analysis_image_size),
        "depth_backend_used": response.depth_backend_used,
        "annotation_backend_used": response.annotation_backend_used,
        "reconstruction_backend_used": response.reconstruction_backend_used,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for index, component in enumerate(response.components, start=1):
        assert component.label
        assert len(component.bbox_2d) == 4
        _assert_object_response(component)
        _save_object_response(component, run_dir, f"component_{index:02d}")

    timing_collector.add(
        "gRPC",
        "Scene components",
        1,
        elapsed,
        str(run_dir),
        task="Scene Components 3D Reconstruction",
        model="Depth + YOLO + RealESRGAN + TripoSR",
        device=os.environ.get("VIZION3D_TEST_RECON_DEVICE", "cpu"),
    )
    assert elapsed < SCENE_LIMIT
