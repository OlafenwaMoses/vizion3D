from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

try:
    import open3d as o3d
    import trimesh

    from vizion3d.proto import lifting_pb2
    from vizion3d.reconstruction import (
        Object3DReconstructionCommand,
        Object3DReconstructionConfig,
        Object3DReconstructionResult,
        SceneComponents3DReconstructionCommand,
        SceneComponents3DReconstructionConfig,
    )
    from vizion3d.reconstruction.defaults import extract_model_bundle, resolve_model_bundle
    from vizion3d.reconstruction.handlers import (
        Object3DReconstructionHandler,
        SceneComponents3DReconstructionHandler,
        _import_tsr_from_bundle,
        _rembg_providers,
    )
    from vizion3d.server.grpc.server import LiftingServiceServicer
    from vizion3d.server.rest.app import create_app
except ModuleNotFoundError as exc:
    optional_modules = {"open3d", "trimesh"}
    if exc.name in optional_modules:
        pytest.skip(f"{exc.name} is required for reconstruction tests", allow_module_level=True)
    raise


def _image_bytes(size=(64, 64), mode="RGBA") -> bytes:
    image = Image.new(mode, size, (80, 120, 160, 255) if mode == "RGBA" else "blue")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _cloud(points=4):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.zeros((points, 3)))
    cloud.colors = o3d.utility.Vector3dVector(np.full((points, 3), 211 / 255))
    return cloud


def _result(point_count=12):
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = np.tile([211, 211, 211, 255], (len(mesh.vertices), 1))
    return Object3DReconstructionResult(
        mesh=mesh,
        point_cloud=_cloud(point_count),
        backend_used="/models",
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        point_count=point_count,
    )


def test_reconstruction_model_bundle_auto_downloads_default_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("VIZION3D_RECONSTRUCTION_MODEL_BUNDLE", raising=False)
    monkeypatch.setenv("VIZION3D_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(
        "vizion3d.reconstruction.defaults.Path.is_file",
        lambda path: False,
    )
    seen = {}

    def download(url, destination):
        seen["url"] = url
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"zip")

    monkeypatch.setattr("vizion3d.reconstruction.defaults._download", download)

    bundle = resolve_model_bundle()

    assert seen["url"].endswith("/scene-components-3d-models.zip")
    assert bundle == tmp_path / "cache" / "scene-components-3d-models.zip"


def test_extract_model_bundle_validates_and_caches_tiny_zip(tmp_path, monkeypatch):
    bundle = tmp_path / "models.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("TripoSR/model.ckpt", b"x")
        archive.writestr("TripoSR/config.yaml", b"x")
        archive.writestr("TripoSR/dino-vitb16-config.json", b"{}")
        archive.writestr("TripoSR/tsr/__init__.py", b"")
        archive.writestr("TripoSR/tsr/system.py", b"class TSR:\n    pass\n")
        archive.writestr("rembg/u2net.onnx", b"x")
        archive.writestr("ESRGAN/RealESRGAN_x4plus.pth", b"x")
    monkeypatch.setenv("VIZION3D_MODEL_CACHE", str(tmp_path / "cache"))

    extracted = extract_model_bundle(bundle)

    assert (extracted / ".complete").is_file()
    assert extract_model_bundle(bundle) == extracted


def test_extract_model_bundle_requires_runtime_source(tmp_path, monkeypatch):
    bundle = tmp_path / "models.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("TripoSR/model.ckpt", b"x")
        archive.writestr("TripoSR/config.yaml", b"x")
        archive.writestr("TripoSR/dino-vitb16-config.json", b"{}")
        archive.writestr("rembg/u2net.onnx", b"x")
        archive.writestr("ESRGAN/RealESRGAN_x4plus.pth", b"x")
    monkeypatch.setenv("VIZION3D_MODEL_CACHE", str(tmp_path / "cache"))

    with pytest.raises(ValueError, match="TripoSR/tsr/system.py"):
        extract_model_bundle(bundle)


def test_extract_model_bundle_requires_enhancement_weights(tmp_path, monkeypatch):
    bundle = tmp_path / "models.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("TripoSR/model.ckpt", b"x")
        archive.writestr("TripoSR/config.yaml", b"x")
        archive.writestr("TripoSR/dino-vitb16-config.json", b"{}")
        archive.writestr("TripoSR/tsr/__init__.py", b"")
        archive.writestr("TripoSR/tsr/system.py", b"class TSR:\n    pass\n")
        archive.writestr("rembg/u2net.onnx", b"x")
    monkeypatch.setenv("VIZION3D_MODEL_CACHE", str(tmp_path / "cache"))

    with pytest.raises(ValueError, match="ESRGAN/RealESRGAN_x4plus.pth"):
        extract_model_bundle(bundle)


def test_import_tsr_from_bundle_uses_extracted_runtime_source(tmp_path):
    source = tmp_path / "TripoSR" / "tsr"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("", encoding="utf-8")
    (source / "system.py").write_text(
        "class TSR:\n    marker = 'bundle'\n",
        encoding="utf-8",
    )

    TSR = _import_tsr_from_bundle(tmp_path)

    assert TSR.marker == "bundle"
    assert str(tmp_path / "TripoSR") == sys.path[0]


def test_object_reconstruction_outputs_gray_mesh_and_cloud(tmp_path):
    class FakeModel:
        def __call__(self, images, device):
            return torch.zeros(1)

        def extract_mesh(self, *args, **kwargs):
            return [trimesh.creation.box()]

    handler = Object3DReconstructionHandler()
    config = Object3DReconstructionConfig(
        point_count=20,
        smoothing_iterations=0,
    )
    with (
        patch.object(
            handler,
            "_load_model",
            return_value=(FakeModel(), torch, trimesh, "cpu", tmp_path),
        ),
        patch.object(
            handler,
            "_prepare_foreground",
            return_value=Image.new("RGB", (512, 512)),
        ),
    ):
        result = handler.handle(
            Object3DReconstructionCommand(image_input=_image_bytes(), advanced_config=config)
        )

    assert result.point_count == 20
    assert np.all(np.asarray(result.mesh.visual.vertex_colors)[:, :3] == 211)
    assert np.allclose(np.asarray(result.point_cloud.colors), 211 / 255)


def test_object_reconstruction_caps_input_at_1080_before_foreground_processing(
    tmp_path,
):
    class FakeModel:
        def __call__(self, images, device):
            return torch.zeros(1)

        def extract_mesh(self, *args, **kwargs):
            return [trimesh.creation.box()]

    seen = {}

    def prepare(image, root, config):
        seen["size"] = image.size
        return Image.new("RGB", (512, 512))

    handler = Object3DReconstructionHandler()
    with (
        patch.object(
            handler,
            "_load_model",
            return_value=(FakeModel(), torch, trimesh, "cpu", tmp_path),
        ),
        patch.object(handler, "_prepare_foreground", side_effect=prepare),
    ):
        handler.handle(
            Object3DReconstructionCommand(
                image_input=_image_bytes((2160, 1080)),
                advanced_config=Object3DReconstructionConfig(
                    point_count=1,
                    smoothing_iterations=0,
                ),
            )
        )

    assert seen["size"] == (1080, 540)


def test_object_foreground_processing_always_uses_rembg(tmp_path, monkeypatch):
    calls = {}

    def new_session(model_name, providers=None):
        calls["model_name"] = model_name
        calls["providers"] = providers
        return "session"

    def remove(image, session):
        calls["image_mode"] = image.mode
        calls["session"] = session
        output = Image.new("RGBA", image.size, (80, 120, 160, 255))
        return output

    monkeypatch.setitem(
        sys.modules,
        "rembg",
        SimpleNamespace(new_session=new_session, remove=remove),
    )

    result = Object3DReconstructionHandler._prepare_foreground(
        Image.new("RGBA", (64, 32), (80, 120, 160, 12)),
        tmp_path,
        Object3DReconstructionConfig(device="cpu"),
    )

    assert calls == {
        "model_name": "u2net",
        "providers": ["CPUExecutionProvider"],
        "image_mode": "RGB",
        "session": "session",
    }
    assert result.size == (512, 512)
    assert result.mode == "RGB"


def test_rembg_provider_selection_prefers_requested_cuda(monkeypatch):
    ort = SimpleNamespace(
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    assert _rembg_providers("cuda") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert _rembg_providers("mps") == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_scene_analysis_limit_does_not_limit_source_crop_resolution():
    original = _image_bytes((2000, 1000))
    depth = SimpleNamespace(point_cloud=_cloud(), backend_used="depth")
    annotation = SimpleNamespace(
        label="chair",
        class_id=56,
        confidence=0.9,
        bbox_2d=[0.0, 0.0, 540.0, 540.0],
        mask_2d=np.ones((540, 1080), dtype=bool),
    )
    annotations = SimpleNamespace(annotations=[annotation], backend_used="annotation")
    seen_crop = {}
    seen_enhancement = {}

    def reconstruct(command):
        seen_crop["size"] = Image.open(io.BytesIO(command.image_input)).size
        return _result()

    def enhance(crop, root, device):
        seen_enhancement["size"] = crop.size
        seen_enhancement["root"] = root
        seen_enhancement["device"] = device
        return crop

    with (
        patch("vizion3d.reconstruction.handlers.DepthEstimation") as depth_task,
        patch("vizion3d.reconstruction.handlers.ObjectMaskAnnotation3D") as annotation_task,
        patch.object(Object3DReconstructionHandler, "handle", side_effect=reconstruct),
        patch(
            "vizion3d.reconstruction.handlers.extract_model_bundle",
            return_value=Path("/models"),
        ),
        patch(
            "vizion3d.reconstruction.handlers._enhance_scene_crop_with_esrgan",
            side_effect=enhance,
        ),
    ):
        depth_task.return_value.run.return_value = depth
        annotation_task.return_value.run.return_value = annotations
        result = SceneComponents3DReconstructionHandler().handle(
            SceneComponents3DReconstructionCommand(
                image_input=original,
                advanced_config=SceneComponents3DReconstructionConfig(
                    max_input_dimension=1080,
                    padding_ratio=0,
                    object_config=Object3DReconstructionConfig(),
                ),
            )
        )

    assert result.source_image_size == (2000, 1000)
    assert result.analysis_image_size == (1080, 540)
    assert seen_enhancement == {
        "size": (1000, 1000),
        "root": Path("/models"),
        "device": "auto",
    }
    assert seen_crop["size"] == (1000, 1000)


def test_reconstruction_rest_endpoint_serializes_geometry():
    client = TestClient(create_app())
    with patch("vizion3d.server.rest.reconstruction.Object3DReconstruction") as task:
        task.return_value.run.return_value = _result()
        response = client.post(
            "/reconstruction/object-3d-reconstruction",
            files={"image": ("object.png", _image_bytes(), "image/png")},
            data={
                "max_input_dimension": "720",
            },
        )

    assert response.status_code == 201
    job_id = response.json()["job_id"]
    for _ in range(20):
        result_response = client.get(f"/reconstruction/object-3d-reconstruction/{job_id}")
        if result_response.status_code != 202:
            break
        time.sleep(0.01)
    assert result_response.status_code == 200
    result = result_response.json()["result"]
    assert result["mesh_ply"]
    assert result["point_cloud_ply"]
    command = task.return_value.run.call_args.args[0]
    assert command.advanced_config.max_input_dimension == 720


def test_reconstruction_grpc_submits_job_and_serializes_geometry():
    request = lifting_pb2.Object3DReconstructionRequest(
        image_bytes=_image_bytes(),
        model_bundle="/models.zip",
        advanced_config=lifting_pb2.Object3DReconstructionConfig(
            point_count=12,
            max_input_dimension=720,
        ),
    )
    with patch("vizion3d.server.grpc.server.Object3DReconstruction") as task:
        task.return_value.run.return_value = _result()
        servicer = LiftingServiceServicer()
        submission = servicer.RunObject3DReconstruction(request, None)
        for _ in range(20):
            response = servicer.GetObject3DReconstructionResult(
                lifting_pb2.ReconstructionJobResultRequest(job_id=submission.job_id),
                None,
            )
            if response.status not in {
                lifting_pb2.RECONSTRUCTION_JOB_STATUS_QUEUED,
                lifting_pb2.RECONSTRUCTION_JOB_STATUS_RUNNING,
            }:
                break
            time.sleep(0.01)

    command = task.return_value.run.call_args.args[0]
    assert command.model_bundle == "/models.zip"
    assert command.advanced_config.point_count == 12
    assert command.advanced_config.max_input_dimension == 720
    assert response.status == lifting_pb2.RECONSTRUCTION_JOB_STATUS_SUCCEEDED
    assert response.result.mesh_ply.startswith(b"ply")
    assert response.result.point_cloud_ply.startswith(b"ply")


def test_reconstruction_configs_reject_invalid_geometry_settings():
    with pytest.raises(ValueError):
        Object3DReconstructionConfig(point_count=0)
    with pytest.raises(ValueError):
        Object3DReconstructionConfig(max_input_dimension=0)
    with pytest.raises(ValueError):
        Object3DReconstructionConfig(max_input_dimension=1081)
    with pytest.raises(ValueError):
        SceneComponents3DReconstructionConfig(confidence_threshold=1.1)
