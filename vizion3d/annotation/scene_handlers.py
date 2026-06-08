"""
CQRS command handler for the SceneMaskAnnotation3D task.

Pipeline
--------
1. Extract XYZ + colour arrays from the input point cloud.
2. Load image — from ``image_input`` (bytes / file path) or by rendering a
   synthetic front-view image from the point cloud itself.
3. Resolve / download the SegFormer-B4 checkpoint.
4. Load and cache the model (thread-safe double-checked locking).
5. Run semantic segmentation → an HxW class-id map (ADE20K, 150 classes).
6. Back-project all 3D points into 2D using camera intrinsics.
7. Group points by the semantic class of the pixel they land on — one
   annotation per class present in the scene.
8. Optionally build per-class extracted clouds and/or an annotated full cloud
   recoloured by the fixed ADE20K palette.

Semantic vs instance
--------------------
This is *semantic* segmentation: every pixel is assigned exactly one of 150
classes, so each 3D point belongs to exactly one class.  Output groups points
by class (all "wall" points together), in contrast to ObjectMaskAnnotation3D
which yields one entry per detected object instance.
"""

from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image

from vizion3d.core.cqrs import CommandHandler

from ._geometry import (
    _backproject,
    _clone_cloud,
    _derive_intrinsics_from_cloud,
    _extract_cloud_arrays,
    _make_full_cloud,
    _make_sub_cloud,
    _render_front_view,
    _resolve_intrinsics,
)
from .commands import SceneMaskAnnotation3DCommand
from .models import SceneMaskAnnotation3DResult, SemanticMaskAnnotation3D
from .scene_defaults import ADE20K_CLASSES, ADE20K_PALETTE, resolve_scene_model_backend


class SceneMaskAnnotation3DHandler(
    CommandHandler[SceneMaskAnnotation3DCommand, SceneMaskAnnotation3DResult]
):
    """Handles :class:`~vizion3d.annotation.commands.SceneMaskAnnotation3DCommand`.

    SegFormer models are loaded on first use and cached in a class-level dict so
    subsequent calls within the same process reuse weights.  A lock ensures
    thread-safe initialisation.
    """

    _scene_models: dict = {}
    _model_lock = threading.Lock()

    def handle(self, command: SceneMaskAnnotation3DCommand) -> SceneMaskAnnotation3DResult:
        model_id = resolve_scene_model_backend(command.model_backend)

        pts, colors_orig = _extract_cloud_arrays(command.point_cloud)

        if command.image_input is None:
            cfg = _derive_intrinsics_from_cloud(pts, command.advanced_config)
            image = _render_front_view(pts, colors_orig, cfg)
        elif isinstance(command.image_input, str):
            image = Image.open(command.image_input).convert("RGB")
            cfg = _resolve_intrinsics(command.advanced_config, *image.size)
        else:
            image = Image.open(io.BytesIO(command.image_input)).convert("RGB")
            cfg = _resolve_intrinsics(command.advanced_config, *image.size)

        img_w, img_h = image.size

        seg = self._run_segformer(model_id, image, cfg)  # (H, W) int class ids

        in_bounds_idx, u_ib, v_ib = _backproject(pts, img_w, img_h, cfg)
        seg_at_points = seg[v_ib, u_ib] if len(in_bounds_idx) else np.empty(0, dtype=seg.dtype)

        colors_new = colors_orig.copy() if command.return_annotated_cloud else None

        annotations: list[SemanticMaskAnnotation3D] = []
        for class_id in np.unique(seg).tolist():
            class_mask = seg == class_id
            ys, xs = np.where(class_mask)
            bbox = (
                [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                if xs.size
                else []
            )

            sel = seg_at_points == class_id
            obj_pt_idx = in_bounds_idx[sel].tolist()
            obj_pt_coords = pts[obj_pt_idx].tolist()

            region_cloud = None
            if command.return_region_clouds and obj_pt_idx:
                import open3d as o3d

                region_cloud = _make_sub_cloud(pts, colors_orig, obj_pt_idx, o3d)

            if colors_new is not None and obj_pt_idx:
                colors_new[obj_pt_idx] = ADE20K_PALETTE[class_id] / 255.0

            annotations.append(
                SemanticMaskAnnotation3D(
                    label=ADE20K_CLASSES[class_id],
                    class_id=int(class_id),
                    bbox_2d=bbox,
                    mask_2d=class_mask,
                    pixel_count=int(class_mask.sum()),
                    point_indices=obj_pt_idx,
                    point_coords=obj_pt_coords,
                    region_cloud=region_cloud,
                )
            )

        annotations.sort(key=lambda a: a.pixel_count, reverse=True)

        annotated_cloud = None
        if command.return_annotated_cloud:
            import open3d as o3d

            cols = colors_new if colors_new is not None else colors_orig
            if len(pts):
                annotated_cloud = _make_full_cloud(pts, cols, o3d)
            else:
                annotated_cloud = _clone_cloud(command.point_cloud, o3d)

        return SceneMaskAnnotation3DResult(
            annotations=annotations,
            annotated_cloud=annotated_cloud,
            backend_used=model_id,
        )

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_segformer(self, model_path: str):
        if model_path in self._scene_models:
            return self._scene_models[model_path]

        with self._model_lock:
            if model_path in self._scene_models:
                return self._scene_models[model_path]

            try:
                import torch

                from .segformer import load_segformer
            except ImportError as exc:
                raise ImportError(
                    "SceneMaskAnnotation3D requires torch. Run: pip install torch"
                ) from exc

            device = torch.device(self._torch_device(torch))
            model = load_segformer(model_path, device)
            self._scene_models[model_path] = (model, device)
            return self._scene_models[model_path]

    # ── Inference ──────────────────────────────────────────────────────────────

    def _run_segformer(self, model_path: str, image: Image.Image, cfg) -> np.ndarray:
        from .segformer import infer_semantic

        model, device = self._load_segformer(model_path)
        return infer_semantic(model, image, device, cfg.inference_size)

    # ── Pre-loading ────────────────────────────────────────────────────────────

    @classmethod
    def preload(cls, model_path: str) -> None:
        resolved = resolve_scene_model_backend(model_path)
        cls()._load_segformer(resolved)

    # ── Device selection ───────────────────────────────────────────────────────

    @staticmethod
    def _torch_device(torch_module) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
