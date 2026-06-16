"""Handlers for object and scene-component 3D reconstruction."""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand
from vizion3d.annotation.defaults import DEFAULT_ANNOTATION_MODEL_URL
from vizion3d.annotation.models import ObjectMaskAnnotation3DConfig
from vizion3d.core.cqrs import CommandHandler
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand
from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_URL

from .commands import (
    Object3DReconstructionCommand,
    SceneComponents3DReconstructionCommand,
)
from .defaults import extract_model_bundle
from .models import (
    Object3DReconstructionResult,
    SceneComponent3D,
    SceneComponents3DReconstructionResult,
)

_GRAY = np.array([211, 211, 211], dtype=np.uint8)
_CPU_DEVICE = "cpu"


def _load_image(value: str | bytes) -> Image.Image:
    if isinstance(value, str):
        image = Image.open(value)
    else:
        image = Image.open(io.BytesIO(value))
    return ImageOps.exif_transpose(image).convert("RGBA")


def _limit_image_dimension(image: Image.Image, maximum: int) -> Image.Image:
    if max(image.size) <= maximum:
        return image
    scale = maximum / max(image.size)
    return image.resize(
        (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )


def _device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _rembg_providers(requested: str) -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    if requested == "auto":
        for provider in (
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "CoreMLExecutionProvider",
        ):
            if provider in available:
                return [provider, "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    requested = requested.lower()
    if requested == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested in {"cuda", "rocm", "amd"} and "ROCMExecutionProvider" in available:
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]
    if requested in {"mps", "coreml"} and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _square_conditioning(rgba: Image.Image, ratio: float) -> Image.Image:
    alpha = np.asarray(rgba.getchannel("A"))
    occupied = np.argwhere(alpha > 8)
    if occupied.size == 0:
        raise ValueError("The input image does not contain a foreground object.")
    y1, x1 = occupied.min(axis=0)
    y2, x2 = occupied.max(axis=0) + 1
    foreground = rgba.crop((x1, y1, x2, y2))
    side = max(foreground.size)
    canvas_side = max(1, int(round(side / ratio)))
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (128, 128, 128, 255))
    canvas.alpha_composite(
        foreground,
        ((canvas_side - foreground.width) // 2, (canvas_side - foreground.height) // 2),
    )
    return canvas.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)


class Object3DReconstructionHandler(
    CommandHandler[Object3DReconstructionCommand, Object3DReconstructionResult]
):
    _models: dict[str, tuple] = {}
    _lock = threading.Lock()

    @classmethod
    def preload(cls, model_bundle: str | None = None) -> None:
        cls()._load_model(model_bundle, "auto")

    def _load_model(self, model_bundle: str | None, requested_device: str):
        root = extract_model_bundle(model_bundle)
        key = f"{root}:{requested_device}"
        if key in self._models:
            return self._models[key]
        with self._lock:
            if key in self._models:
                return self._models[key]
            try:
                import torch
                import trimesh
            except ImportError as exc:
                raise ImportError(
                    "3D reconstruction requires torch, trimesh, einops, omegaconf, "
                    "and PyMCubes."
                ) from exc

            triposr = root / "TripoSR"
            os.environ["TRIPOSR_DINO_CONFIG"] = str(
                triposr / "dino-vitb16-config.json"
            )
            try:
                from tsr.system import TSR
            except ImportError:
                source = os.environ.get("VIZION3D_TRIPOSR_SOURCE")
                if source is None:
                    checkout = (
                        Path(__file__).resolve().parents[2]
                        / "research"
                        / "3D_Object-Reconstruction"
                        / "TripoSR"
                    )
                    source = str(checkout)
                if source not in sys.path:
                    sys.path.insert(0, source)
                try:
                    from tsr.system import TSR
                except ImportError as exc:
                    raise ImportError(
                        "TripoSR runtime source is required. Set "
                        "VIZION3D_TRIPOSR_SOURCE to the TripoSR checkout."
                    ) from exc

            device = _device(torch, requested_device)
            try:
                model = TSR.from_pretrained(
                    str(triposr), config_name="config.yaml", weight_name="model.ckpt"
                )
                model.renderer.set_chunk_size(8192)
                model.to(device).eval()
            except Exception:
                if device == _CPU_DEVICE:
                    raise
                device = _CPU_DEVICE
                model = TSR.from_pretrained(
                    str(triposr), config_name="config.yaml", weight_name="model.ckpt"
                )
                model.renderer.set_chunk_size(8192)
                model.to(device).eval()
            self._models[key] = (model, torch, trimesh, device, root)
            return self._models[key]

    @staticmethod
    def _prepare_foreground(image: Image.Image, root: Path, config) -> Image.Image:
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise ImportError("3D reconstruction requires rembg.") from exc
        os.environ["U2NET_HOME"] = str(root / "rembg")
        try:
            session = new_session("u2net", providers=_rembg_providers(config.device))
            rgba = remove(image.convert("RGB"), session=session).convert("RGBA")
        except Exception:
            session = new_session("u2net", providers=["CPUExecutionProvider"])
            rgba = remove(image.convert("RGB"), session=session).convert("RGBA")

        return _square_conditioning(rgba, config.foreground_ratio)

    def handle(self, command: Object3DReconstructionCommand):
        config = command.advanced_config
        model, torch, trimesh, device, root = self._load_model(
            command.model_bundle, config.device
        )
        image = _limit_image_dimension(
            _load_image(command.image_input), config.max_input_dimension
        )
        conditioning = self._prepare_foreground(image, root, config)
        try:
            mesh = self._extract_mesh(model, torch, conditioning, device, config)
        except Exception:
            if device == _CPU_DEVICE:
                raise
            model, torch, trimesh, device, root = self._load_model(
                command.model_bundle, _CPU_DEVICE
            )
            mesh = self._extract_mesh(model, torch, conditioning, device, config)

        return self._postprocess_mesh(mesh, torch, trimesh, root, config)

    @staticmethod
    def _extract_mesh(model, torch, conditioning, device: str, config):
        autocast = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if device == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            scene_codes = model([conditioning], device=device)
            mesh = model.extract_mesh(
                scene_codes,
                has_vertex_color=False,
                resolution=config.marching_cubes_resolution,
                threshold=config.density_threshold,
            )[0]
        return mesh

    @staticmethod
    def _postprocess_mesh(mesh, torch, trimesh, root: Path, config):
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        components = list(mesh.split(only_watertight=False))
        if components:
            largest = max(component.area for component in components)
            mesh = trimesh.util.concatenate(
                [
                    component
                    for component in components
                    if component.area >= largest * config.min_component_area_ratio
                ]
            )
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        if config.smoothing_iterations:
            from trimesh.smoothing import filter_taubin

            filter_taubin(
                mesh,
                lamb=0.35,
                nu=0.36,
                iterations=config.smoothing_iterations,
            )
        colors = np.tile(np.append(_GRAY, 255), (len(mesh.vertices), 1))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)

        points, _ = trimesh.sample.sample_surface(
            mesh, max(1, config.point_count), seed=42
        )
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(
            np.tile(_GRAY / 255.0, (len(points), 1))
        )
        return Object3DReconstructionResult(
            mesh=mesh,
            point_cloud=cloud,
            backend_used=str(root),
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            point_count=len(points),
        )


def _padded_square(bbox, size, ratio):
    width, height = size
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1, y2 - y1, 1.0) * (1.0 + 2.0 * ratio)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (
        max(0, math.floor(cx - side / 2)),
        max(0, math.floor(cy - side / 2)),
        min(width, math.ceil(cx + side / 2)),
        min(height, math.ceil(cy + side / 2)),
    )


def _enhance_scene_crop_with_esrgan(
    rgba: Image.Image, root: Path, requested_device: str
) -> Image.Image:
    try:
        import cv2
        import torch
    except ImportError as exc:
        raise ImportError(
            "SceneComponents3DReconstruction requires basicsr, realesrgan, "
            "and opencv-python for Real-ESRGAN crop enhancement."
        ) from exc

    model_path = root / "ESRGAN" / "RealESRGAN_x4plus.pth"
    if not model_path.is_file():
        raise ValueError(
            "Model bundle is missing ESRGAN/RealESRGAN_x4plus.pth, required by "
            "SceneComponents3DReconstruction."
        )

    device = _device(torch, requested_device)
    working = rgba.copy()
    working.thumbnail((512, 512), Image.Resampling.LANCZOS)
    try:
        return _run_esrgan_crop(
            rgba=working, root=root, cv2=cv2, torch=torch, device=device
        )
    except Exception:
        if device == _CPU_DEVICE:
            raise
        return _run_esrgan_crop(
            rgba=working, root=root, cv2=cv2, torch=torch, device=_CPU_DEVICE
        )


def _run_esrgan_crop(
    rgba: Image.Image, root: Path, cv2, torch, device: str
) -> Image.Image:
    _ensure_torchvision_functional_tensor_compat()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:
        raise ImportError(
            "SceneComponents3DReconstruction requires basicsr and realesrgan "
            "for Real-ESRGAN crop enhancement."
        ) from exc

    model_path = root / "ESRGAN" / "RealESRGAN_x4plus.pth"
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=256,
        tile_pad=16,
        pre_pad=0,
        half=device == "cuda",
        device=torch.device(device),
    )
    rgb = np.asarray(rgba.convert("RGB"))
    enhanced, _ = upsampler.enhance(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), outscale=2
    )
    enhanced_rgb = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    alpha = rgba.getchannel("A").resize(enhanced_rgb.size, Image.Resampling.LANCZOS)
    result = enhanced_rgb.convert("RGBA")
    result.putalpha(alpha)
    return result


def _ensure_torchvision_functional_tensor_compat() -> None:
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        from torchvision.transforms import functional as functional_tensor
    except ImportError:
        return
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor


class SceneComponents3DReconstructionHandler(
    CommandHandler[
        SceneComponents3DReconstructionCommand,
        SceneComponents3DReconstructionResult,
    ]
):
    def handle(self, command: SceneComponents3DReconstructionCommand):
        config = command.advanced_config
        original = _load_image(command.image_input)
        analysis = original.convert("RGB")
        if (
            config.max_input_dimension > 0
            and max(analysis.size) > config.max_input_dimension
        ):
            scale = config.max_input_dimension / max(analysis.size)
            analysis = analysis.resize(
                (
                    max(1, round(analysis.width * scale)),
                    max(1, round(analysis.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        analysis.save(buffer, format="PNG")
        analysis_bytes = buffer.getvalue()
        depth = DepthEstimation().run(
            DepthEstimationCommand(
                image_input=analysis_bytes,
                model_backend=command.depth_model_backend or DEFAULT_DEPTH_MODEL_URL,
                return_depth_image=False,
                return_raw_depth=False,
                return_point_cloud=True,
            )
        )
        if depth.point_cloud is None:
            raise RuntimeError("Depth estimation did not produce a point cloud.")
        annotations = ObjectMaskAnnotation3D().run(
            ObjectMaskAnnotation3DCommand(
                point_cloud=depth.point_cloud,
                image_input=analysis_bytes,
                model_backend=(
                    command.annotation_model_backend or DEFAULT_ANNOTATION_MODEL_URL
                ),
                advanced_config=ObjectMaskAnnotation3DConfig(
                    conf_threshold=config.confidence_threshold
                ),
            )
        )
        selected = annotations.annotations
        if config.max_objects > 0:
            selected = selected[: config.max_objects]

        components = []
        object_handler = Object3DReconstructionHandler()
        reconstruction_root = extract_model_bundle(command.model_bundle)
        sx = original.width / analysis.width
        sy = original.height / analysis.height
        for annotation in selected:
            mask = Image.fromarray(
                np.asarray(annotation.mask_2d, dtype=np.uint8) * 255, mode="L"
            )
            mask = mask.filter(ImageFilter.MaxFilter(3)).filter(
                ImageFilter.MinFilter(3)
            )
            box = _padded_square(annotation.bbox_2d, analysis.size, config.padding_ratio)
            source_box = (
                math.floor(box[0] * sx),
                math.floor(box[1] * sy),
                math.ceil(box[2] * sx),
                math.ceil(box[3] * sy),
            )
            source_mask = mask.resize(original.size, Image.Resampling.NEAREST)
            crop = original.crop(source_box)
            crop.putalpha(source_mask.crop(source_box))
            crop = _enhance_scene_crop_with_esrgan(
                crop, reconstruction_root, config.object_config.device
            )
            crop_buffer = io.BytesIO()
            crop.save(crop_buffer, format="PNG")
            result = object_handler.handle(
                Object3DReconstructionCommand(
                    image_input=crop_buffer.getvalue(),
                    model_bundle=command.model_bundle,
                    advanced_config=config.object_config,
                )
            )
            components.append(
                SceneComponent3D(
                    label=annotation.label,
                    class_id=annotation.class_id,
                    confidence=annotation.confidence,
                    bbox_2d=annotation.bbox_2d,
                    mesh=result.mesh,
                    point_cloud=result.point_cloud,
                    vertex_count=result.vertex_count,
                    face_count=result.face_count,
                    point_count=result.point_count,
                )
            )
        return SceneComponents3DReconstructionResult(
            components=components,
            source_image_size=original.size,
            analysis_image_size=analysis.size,
            depth_backend_used=depth.backend_used,
            annotation_backend_used=annotations.backend_used,
            reconstruction_backend_used=str(reconstruction_root),
        )
