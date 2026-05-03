"""
CQRS command handler for the Stereo Depth task.

Orchestrates model loading, image pre-processing, stereo inference, depth
computation, and optional point-cloud / mesh generation.
"""

from __future__ import annotations

import contextlib
import io
import threading

import numpy as np
from PIL import Image

from vizion3d.core.cqrs import CommandHandler
from vizion3d.lifting.handlers import (
    OPEN3D_CAMERA_TO_IMAGE_VIEW_TRANSFORM,
    DepthEstimationHandler,
)

from .arch import build_s2m2
from .arch.utils import image_crop, image_pad
from .commands import StereoDepthCommand
from .defaults import resolve_stereo_model_backend
from .models import StereoDepthResult

# Flip-Y transform reused from the depth estimation handler so point-cloud
# orientation matches the image-plane convention (Y increases downward).
_CAMERA_TO_IMAGE_VIEW = OPEN3D_CAMERA_TO_IMAGE_VIEW_TRANSFORM


class StereoDepthHandler(CommandHandler[StereoDepthCommand, StereoDepthResult]):
    """Handles :class:`~vizion3d.stereo.commands.StereoDepthCommand` inference requests.

    Models are loaded on first use and cached in a class-level dict so subsequent
    calls within the same process reuse weights without re-reading disk.  A lock
    ensures thread-safe initialisation.

    The inference pipeline:
    1. Resolve / download the checkpoint.
    2. Load and cache the S2M2 model (thread-safe).
    3. Pre-process the image pair (resize, pad to 32-px multiple).
    4. Run stereo inference with mixed-precision autocast where available.
    5. Crop output back to original resolution; optionally upsample if scaled.
    6. Convert disparity → metric depth using the pinhole stereo formula.
    7. Unproject depth + colour into an Open3D PointCloud and optionally mesh.
    """

    _stereo_models: dict = {}
    _model_lock = threading.Lock()

    def handle(self, command: StereoDepthCommand) -> StereoDepthResult:
        """Execute the full stereo depth inference pipeline.

        Args:
            command: Fully populated :class:`StereoDepthCommand`.

        Returns:
            :class:`StereoDepthResult` with depth map, disparity map, and
            optional depth image, point cloud, and mesh.
        """
        model_id = resolve_stereo_model_backend(command.model_backend)

        def _load_image(src: str | bytes) -> Image.Image:
            if isinstance(src, str):
                return Image.open(src).convert("RGB")
            return Image.open(io.BytesIO(src)).convert("RGB")

        left_pil = _load_image(command.left_image)
        right_pil = _load_image(command.right_image)

        inference = self._run_s2m2(model_id, left_pil, right_pil, command.advanced_config)
        if isinstance(inference, tuple):
            disp_np, occ_np, conf_np = inference
        else:
            # Backwards-compatible path for tests or custom subclasses that patch
            # _run_s2m2 to return only a disparity map.
            disp_np, occ_np, conf_np = inference, None, None

        cfg = command.advanced_config

        # Disparity → metric depth (millimetre formula → metres)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_mm = cfg.baseline * cfg.focal_length / (disp_np + cfg.doffs)
        depth_mm[disp_np <= 0] = 0.0
        depth_m = depth_mm / 1000.0

        min_depth = float(np.min(depth_m))
        max_depth = float(np.max(depth_m))
        depth_map: list[list[float]] = depth_m.astype(np.float32).tolist()
        disparity_map: list[list[float]] = disp_np.astype(np.float32).tolist()

        depth_image = None
        if command.return_depth_image:
            try:
                import open3d as o3d
            except ImportError:
                raise ImportError(
                    "open3d is required for depth image output. Pin to Python 3.12 and run: uv sync"
                )
            depth_range = max_depth - min_depth
            if depth_range > 0:
                normalized = (depth_m - min_depth) / depth_range
            else:
                normalized = np.zeros_like(depth_m)
            depth_16bit = (normalized * 65535).astype(np.uint16)
            depth_image = o3d.geometry.Image(depth_16bit)

        point_cloud = None
        mesh = None
        if command.return_point_cloud or command.return_mesh:
            try:
                import open3d as o3d
            except ImportError:
                raise ImportError(
                    "open3d is required for point cloud / mesh output. "
                    "Pin to Python 3.12 and run: uv sync"
                )
            point_cloud, mesh = self._unproject(
                left_pil,
                depth_m,
                disp_np,
                cfg,
                o3d,
                occ_np=occ_np,
                conf_np=conf_np,
                want_cloud=command.return_point_cloud,
                want_mesh=command.return_mesh,
            )

        return StereoDepthResult(
            depth_map=depth_map,
            disparity_map=disparity_map,
            min_depth=min_depth,
            max_depth=max_depth,
            backend_used=model_id,
            depth_image=depth_image,
            point_cloud=point_cloud,
            mesh=mesh,
            point_cloud_scale=1.0,
        )

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_s2m2(self, model_path: str):
        """Load and cache an S2M2 checkpoint (thread-safe double-checked locking).

        Args:
            model_path: Resolved local path to the ``.pth`` checkpoint.

        Returns:
            ``(model, torch, device)`` tuple ready for inference.
        """
        if model_path in self._stereo_models:
            return self._stereo_models[model_path]

        with self._model_lock:
            if model_path in self._stereo_models:
                return self._stereo_models[model_path]

            try:
                import torch
            except ImportError as exc:
                raise ImportError("S2M2 stereo checkpoints require torch. Run: uv sync") from exc

            device = self._torch_device(torch)
            model = build_s2m2(model_path)

            try:
                ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                ckpt = torch.load(model_path, map_location="cpu")

            state_dict = ckpt.get("state_dict", ckpt)
            model.my_load_state_dict(state_dict)
            model = model.to(device).eval()

            self._stereo_models[model_path] = (model, torch, device)
            return self._stereo_models[model_path]

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_s2m2(
        self,
        model_path: str,
        left: Image.Image,
        right: Image.Image,
        cfg,
    ) -> np.ndarray:
        """Run one stereo forward pass and return the disparity map (in pixels).

        Handles input scaling, padding, autocast, output cropping, and upsampling.

        Args:
            model_path: Resolved local path to the checkpoint.
            left: Left PIL image (RGB).
            right: Right PIL image (RGB).
            cfg: :class:`~vizion3d.stereo.models.StereoDepthAdvancedConfig`.

        Returns:
            Tuple of ``(disparity, occlusion, confidence)`` float32 numpy arrays,
            each shape ``(H, W)`` at the original image resolution.
        """
        model, torch, device = self._load_s2m2(model_path)

        import torch.nn.functional as F

        img_h, img_w = left.height, left.width

        left_t = torch.from_numpy(np.asarray(left).copy()).permute(2, 0, 1).unsqueeze(0).float()
        right_t = torch.from_numpy(np.asarray(right).copy()).permute(2, 0, 1).unsqueeze(0).float()

        sf = cfg.scale_factor
        if sf != 1.0:
            scaled_h = round(img_h * sf / 32) * 32
            scaled_w = round(img_w * sf / 32) * 32
            left_t_inp = F.interpolate(
                left_t,
                (scaled_h, scaled_w),
                mode="bilinear",
                align_corners=False,
            )
            right_t_inp = F.interpolate(
                right_t,
                (scaled_h, scaled_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            left_t_inp = left_t
            right_t_inp = right_t

        left_pad = image_pad(left_t_inp, 32).to(device)
        right_pad = image_pad(right_t_inp, 32).to(device)

        device_type = device if isinstance(device, str) else device.type
        if device_type == "cuda":
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        elif device_type == "mps":
            autocast_ctx = torch.amp.autocast(device_type="mps", dtype=torch.float16, enabled=True)
        else:
            autocast_ctx = contextlib.nullcontext()

        with torch.inference_mode(), autocast_ctx:
            pred_disp, pred_occ, pred_conf = model(left_pad, right_pad)

        if sf != 1.0:
            scaled_h = round(img_h * sf / 32) * 32
            scaled_w = round(img_w * sf / 32) * 32
            pred_disp = image_crop(pred_disp, (scaled_h, scaled_w))
            pred_disp = (
                F.interpolate(
                    pred_disp,
                    (img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                )
                / sf
            )
            pred_occ = image_crop(pred_occ, (scaled_h, scaled_w))
            pred_occ = F.interpolate(
                pred_occ,
                (img_h, img_w),
                mode="bilinear",
                align_corners=False,
            )
            pred_conf = image_crop(pred_conf, (scaled_h, scaled_w))
            pred_conf = F.interpolate(
                pred_conf,
                (img_h, img_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            pred_disp = image_crop(pred_disp, (img_h, img_w))
            pred_occ = image_crop(pred_occ, (img_h, img_w))
            pred_conf = image_crop(pred_conf, (img_h, img_w))

        if device_type == "mps":
            torch.mps.empty_cache()

        return (
            pred_disp.squeeze().float().cpu().numpy(),
            pred_occ.squeeze().float().cpu().numpy(),
            pred_conf.squeeze().float().cpu().numpy(),
        )

    # ── Point cloud / mesh ────────────────────────────────────────────────────

    def _unproject(
        self,
        left_pil,
        depth_m,
        disp_np,
        cfg,
        o3d,
        *,
        occ_np=None,
        conf_np=None,
        want_cloud,
        want_mesh,
    ):
        """Unproject the left image into a coloured 3D point cloud and optional mesh.

        Args:
            left_pil: Left PIL image (RGB) — used for vertex colours.
            depth_m: Metric depth map, shape ``(H, W)``.
            disp_np: Raw disparity map, shape ``(H, W)``.
            cfg: :class:`~vizion3d.stereo.models.StereoDepthAdvancedConfig`.
            o3d: Imported open3d module.
            occ_np: Optional per-pixel occlusion score, shape ``(H, W)``.
            conf_np: Optional per-pixel confidence score, shape ``(H, W)``.
            want_cloud: Whether to return the point cloud.
            want_mesh: Whether to return the mesh.

        Returns:
            ``(point_cloud, mesh)`` — each is the Open3D object or ``None`` if not requested.
        """
        H, W = depth_m.shape
        left_np = np.asarray(left_pil)

        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        x = (uu - cfg.cx) * depth_m / cfg.focal_length
        y = (vv - cfg.cy) * depth_m / cfg.focal_length
        z = depth_m

        valid = (z > 0) & (z < cfg.z_far)
        if conf_np is not None:
            valid &= conf_np >= cfg.conf_threshold
        if occ_np is not None:
            valid &= occ_np >= cfg.occ_threshold
        pts = np.stack([x, y, z], axis=-1)[valid]
        cols = left_np[valid].astype(np.float64) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        # Flip Y so Y increases downward in the viewer, matching image-plane orientation.
        pcd.transform(_CAMERA_TO_IMAGE_VIEW)

        mesh = None
        if want_mesh:
            mesh = DepthEstimationHandler._mesh_from_point_cloud(pcd, o3d)

        return (pcd if want_cloud else None), mesh

    # ── Pre-loading ───────────────────────────────────────────────────────────

    @classmethod
    def preload(cls, model_path: str) -> None:
        """Resolve *model_path* (downloading if a URL) and load it into the class-level cache.

        Call this at server startup to ensure the model is in memory before the first request.
        """
        from .defaults import resolve_stereo_model_backend

        resolved = resolve_stereo_model_backend(model_path)
        cls()._load_s2m2(resolved)

    # ── Device selection ──────────────────────────────────────────────────────

    @staticmethod
    def _torch_device(torch_module) -> str:
        """Return the best available device string: ``'cuda'``, ``'mps'``, or ``'cpu'``."""
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
