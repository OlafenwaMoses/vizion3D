"""
Integration tests — direct Python entry points for 3D reconstruction.

These tests run real image bytes through the new reconstruction tasks. They use
low-cost mesh settings so the path is exercised without the full production
point-count cost.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

for module_name in ("open3d", "trimesh", "rembg", "mcubes", "omegaconf", "einops"):
    assert importlib.util.find_spec(module_name), f"{module_name} is required"

from vizion3d.lifting.utils import create_ply_binary  # noqa: E402
from vizion3d.reconstruction import (  # noqa: E402
    Object3DReconstruction,
    Object3DReconstructionCommand,
    Object3DReconstructionConfig,
    SceneComponents3DReconstruction,
    SceneComponents3DReconstructionCommand,
    SceneComponents3DReconstructionConfig,
)
from vizion3d.reconstruction.defaults import resolve_model_bundle  # noqa: E402
from vizion3d.reconstruction.handlers import Object3DReconstructionHandler  # noqa: E402

OBJECT_LIMIT = float(os.environ.get("VIZION3D_TEST_RECON_OBJECT_LIMIT", "240.0"))
SCENE_LIMIT = float(os.environ.get("VIZION3D_TEST_RECON_SCENE_LIMIT", "420.0"))
TEST_IMAGE_MAX_DIMENSION = 1080
RECONSTRUCTION_IMAGE = "reconstruction_scene_1080.jpg"


@pytest.fixture(scope="session")
def reconstruction_model_bundle() -> str:
    return str(resolve_model_bundle())


@pytest.fixture(scope="session")
def reconstruction_image_bytes() -> bytes:
    path = Path(__file__).parent.parent / "assets" / RECONSTRUCTION_IMAGE
    assert path.is_file(), f"Reconstruction integration image not found: {path}"
    image = Image.open(path)
    assert max(image.size) <= TEST_IMAGE_MAX_DIMENSION
    return path.read_bytes()


def _save_mesh_and_cloud(result, run_dir: Path, stem: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = run_dir / f"{stem}_mesh.ply"
    cloud_path = run_dir / f"{stem}_point_cloud.ply"
    mesh_path.write_bytes(result.mesh.export(file_type="ply", encoding="binary_little_endian"))
    points = np.asarray(result.point_cloud.points).astype(np.float32)
    colors = (np.asarray(result.point_cloud.colors) * 255).astype(np.uint8)
    cloud_path.write_bytes(create_ply_binary(points, colors))


def _assert_object_result(result) -> None:
    assert result.vertex_count > 0
    assert result.face_count > 0
    assert result.point_count > 0
    assert result.mesh.vertices.shape[0] == result.vertex_count
    assert result.mesh.faces.shape[0] == result.face_count
    assert result.point_cloud.has_points()
    assert len(np.asarray(result.point_cloud.points)) == result.point_count
    assert np.all(np.asarray(result.mesh.visual.vertex_colors)[:, :3] == 211)
    assert np.allclose(np.asarray(result.point_cloud.colors), 211 / 255)


def test_object_3d_reconstruction_runs_real_image(
    reconstruction_image_bytes,
    reconstruction_model_bundle,
    tmp_path,
    timing_collector,
):
    image = Image.open(BytesIO(reconstruction_image_bytes))
    assert max(image.size) <= TEST_IMAGE_MAX_DIMENSION
    Object3DReconstructionHandler._models.clear()
    config = Object3DReconstructionConfig(
        max_input_dimension=512,
        marching_cubes_resolution=64,
        point_count=2_048,
        smoothing_iterations=0,
        device=os.environ.get("VIZION3D_TEST_RECON_DEVICE", "cpu"),
    )

    t0 = time.perf_counter()
    result = Object3DReconstruction().run(
        Object3DReconstructionCommand(
            image_input=reconstruction_image_bytes,
            model_bundle=reconstruction_model_bundle,
            advanced_config=config,
        )
    )
    elapsed = time.perf_counter() - t0

    _assert_object_result(result)
    _save_mesh_and_cloud(result, tmp_path / "object_3d_reconstruction", "object")
    timing_collector.add(
        "Direct",
        "TripoSR object",
        1,
        elapsed,
        str(tmp_path / "object_3d_reconstruction"),
        task="Object 3D Reconstruction",
        model="TripoSR",
        device=config.device,
    )
    assert elapsed < OBJECT_LIMIT


def test_scene_components_3d_reconstruction_runs_real_image(
    reconstruction_image_bytes,
    reconstruction_model_bundle,
    local_model_path,
    local_annotation_model_path,
    tmp_path,
    timing_collector,
):
    image = Image.open(BytesIO(reconstruction_image_bytes))
    assert max(image.size) <= TEST_IMAGE_MAX_DIMENSION
    Object3DReconstructionHandler._models.clear()
    object_config = Object3DReconstructionConfig(
        max_input_dimension=512,
        marching_cubes_resolution=64,
        point_count=1_024,
        smoothing_iterations=0,
        device=os.environ.get("VIZION3D_TEST_RECON_DEVICE", "cpu"),
    )
    config = SceneComponents3DReconstructionConfig(
        max_input_dimension=640,
        max_objects=1,
        confidence_threshold=0.01,
        padding_ratio=0.1,
        object_config=object_config,
    )

    t0 = time.perf_counter()
    result = SceneComponents3DReconstruction().run(
        SceneComponents3DReconstructionCommand(
            image_input=reconstruction_image_bytes,
            model_bundle=reconstruction_model_bundle,
            depth_model_backend=local_model_path,
            annotation_model_backend=local_annotation_model_path,
            advanced_config=config,
        )
    )
    elapsed = time.perf_counter() - t0

    assert result.source_image_size[0] > 0 and result.source_image_size[1] > 0
    assert max(result.source_image_size) <= TEST_IMAGE_MAX_DIMENSION
    assert max(result.analysis_image_size) <= config.max_input_dimension
    assert result.depth_backend_used
    assert result.annotation_backend_used
    assert result.reconstruction_backend_used
    assert len(result.components) >= 1

    run_dir = tmp_path / "scene_components_3d_reconstruction"
    summary = {
        "source_image_size": result.source_image_size,
        "analysis_image_size": result.analysis_image_size,
        "components": [
            {
                "label": component.label,
                "class_id": component.class_id,
                "confidence": component.confidence,
                "vertex_count": component.vertex_count,
                "face_count": component.face_count,
                "point_count": component.point_count,
            }
            for component in result.components
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    for index, component in enumerate(result.components, start=1):
        _assert_object_result(component)
        _save_mesh_and_cloud(component, run_dir, f"component_{index:02d}")

    timing_collector.add(
        "Direct",
        "Scene components",
        1,
        elapsed,
        str(run_dir),
        task="Scene Components 3D Reconstruction",
        model="Depth + YOLO + RealESRGAN + TripoSR",
        device=object_config.device,
    )
    assert elapsed < SCENE_LIMIT
