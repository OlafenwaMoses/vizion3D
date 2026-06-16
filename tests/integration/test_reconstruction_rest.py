"""
Integration tests — FastAPI REST entry points for 3D reconstruction.

These tests run the real REST endpoints with the saved 480px reconstruction
image and low-cost mesh settings.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("open3d", reason="open3d required")
pytest.importorskip("trimesh", reason="trimesh required")
pytest.importorskip("rembg", reason="rembg required")
pytest.importorskip("mcubes", reason="PyMCubes required")
pytest.importorskip("omegaconf", reason="omegaconf required")
pytest.importorskip("einops", reason="einops required")

from vizion3d.reconstruction.handlers import Object3DReconstructionHandler  # noqa: E402
from vizion3d.server.rest.app import app  # noqa: E402

OBJECT_LIMIT = float(os.environ.get("VIZION3D_TEST_RECON_REST_OBJECT_LIMIT", "260.0"))
SCENE_LIMIT = float(os.environ.get("VIZION3D_TEST_RECON_REST_SCENE_LIMIT", "460.0"))
TEST_IMAGE_MAX_DIMENSION = 480
RECONSTRUCTION_IMAGE = "reconstruction_scene_480.jpg"

client = TestClient(app, raise_server_exceptions=True)


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


def _decode_ply(value: str) -> bytes:
    ply = base64.b64decode(value)
    assert ply.startswith(b"ply\n")
    return ply


def _save_object_payload(data: dict, run_dir: Path, stem: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{stem}_mesh.ply").write_bytes(_decode_ply(data["mesh_ply"]))
    (run_dir / f"{stem}_point_cloud.ply").write_bytes(
        _decode_ply(data["point_cloud_ply"])
    )


def _assert_object_payload(data: dict) -> None:
    assert data["vertex_count"] > 0
    assert data["face_count"] > 0
    assert data["point_count"] > 0
    assert data["backend_used"]
    _decode_ply(data["mesh_ply"])
    _decode_ply(data["point_cloud_ply"])


def _object_form(model_bundle: str) -> dict[str, str]:
    return {
        "model_bundle": model_bundle,
        "max_input_dimension": "512",
        "marching_cubes_resolution": "64",
        "point_count": "1024",
        "smoothing_iterations": "0",
        "device": os.environ.get("VIZION3D_TEST_RECON_DEVICE", "cpu"),
    }


def test_rest_object_3d_reconstruction_runs_real_image(
    reconstruction_image_bytes,
    reconstruction_model_bundle,
    tmp_path,
    timing_collector,
):
    Object3DReconstructionHandler._models.clear()

    t0 = time.perf_counter()
    response = client.post(
        "/reconstruction/object-3d-reconstruction",
        files={"image": (RECONSTRUCTION_IMAGE, reconstruction_image_bytes, "image/jpeg")},
        data=_object_form(reconstruction_model_bundle),
    )
    elapsed = time.perf_counter() - t0

    assert response.status_code == 200, response.text[:300]
    data = response.json()
    _assert_object_payload(data)
    run_dir = tmp_path / "rest_object_3d_reconstruction"
    _save_object_payload(data, run_dir, "object")
    timing_collector.add(
        "REST",
        "TripoSR object",
        1,
        elapsed,
        str(run_dir),
        task="Object 3D Reconstruction",
        model="TripoSR",
        device=os.environ.get("VIZION3D_TEST_RECON_DEVICE", "cpu"),
    )
    assert elapsed < OBJECT_LIMIT


def test_rest_scene_components_3d_reconstruction_runs_real_image(
    reconstruction_image_bytes,
    reconstruction_model_bundle,
    local_model_path,
    local_annotation_model_path,
    tmp_path,
    timing_collector,
):
    Object3DReconstructionHandler._models.clear()

    t0 = time.perf_counter()
    response = client.post(
        "/reconstruction/scene-components-3d-reconstruction",
        files={"image": (RECONSTRUCTION_IMAGE, reconstruction_image_bytes, "image/jpeg")},
        data={
            **_object_form(reconstruction_model_bundle),
            "depth_model_backend": local_model_path,
            "annotation_model_backend": local_annotation_model_path,
            "max_input_dimension": "640",
            "max_objects": "1",
            "confidence_threshold": "0.05",
            "padding_ratio": "0.1",
        },
    )
    elapsed = time.perf_counter() - t0

    assert response.status_code == 200, response.text[:300]
    data = response.json()
    assert data["source_image_size"]
    assert max(data["source_image_size"]) <= TEST_IMAGE_MAX_DIMENSION
    assert max(data["analysis_image_size"]) <= 640
    assert data["depth_backend_used"]
    assert data["annotation_backend_used"]
    assert data["reconstruction_backend_used"]
    assert len(data["components"]) >= 1

    run_dir = tmp_path / "rest_scene_components_3d_reconstruction"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        key: data[key]
        for key in (
            "source_image_size",
            "analysis_image_size",
            "depth_backend_used",
            "annotation_backend_used",
            "reconstruction_backend_used",
        )
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for index, component in enumerate(data["components"], start=1):
        assert component["label"]
        assert len(component["bbox_2d"]) == 4
        _assert_object_payload(
            {
                **component,
                "backend_used": data["reconstruction_backend_used"],
            }
        )
        _save_object_payload(component, run_dir, f"component_{index:02d}")

    timing_collector.add(
        "REST",
        "Scene components",
        1,
        elapsed,
        str(run_dir),
        task="Scene Components 3D Reconstruction",
        model="Depth + YOLO + RealESRGAN + TripoSR",
        device=os.environ.get("VIZION3D_TEST_RECON_DEVICE", "cpu"),
    )
    assert elapsed < SCENE_LIMIT
