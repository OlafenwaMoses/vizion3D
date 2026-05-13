import contextlib
import io
import math
import threading

import numpy as np
from PIL import Image

from vizion3d.core.cqrs import CommandHandler

from .commands import DepthEstimationCommand
from .defaults import resolve_model_backend
from .depth_anything import convert_depth_anything_v2_state_dict, depth_anything_v2_config
from .models import DepthEstimationResult

# Internal depth parameters for point cloud generation — not user-configurable.
# _DEPTH_TRUNC sets the coordinate scale (metres); _DEPTH_SCALE fills the full
# uint16 range at that scale for maximum depth precision.
# _MIN_DEPTH_M prevents close pixels from collapsing to Z≈0: monocular depth is
# unreliable very close to the lens, and Z≈0 causes X,Y≈0 for all close pixels,
# destroying foreground geometry in the point cloud.
_DEPTH_TRUNC: float = 10.0
_DEPTH_SCALE: int = math.floor(65535 / _DEPTH_TRUNC)  # 6553
_MIN_DEPTH_M: float = 0.5


class DepthEstimationHandler(CommandHandler[DepthEstimationCommand, DepthEstimationResult]):
    _depth_anything_models = {}
    _model_lock = threading.Lock()

    def handle(self, command: DepthEstimationCommand) -> DepthEstimationResult:
        model_id = resolve_model_backend(command.model_backend)

        if isinstance(command.image_input, str):
            image = Image.open(command.image_input).convert("RGB")
        else:
            image = Image.open(io.BytesIO(command.image_input)).convert("RGB")

        depth_array = self._run_depth_anything_checkpoint(model_id, image)

        min_depth = float(np.min(depth_array))
        max_depth = float(np.max(depth_array))
        depth_map = depth_array.astype(np.float32).tolist()

        raw_depth = depth_array.copy() if command.return_raw_depth else None

        depth_image = None
        if command.return_depth_image:
            try:
                import open3d as o3d
            except ImportError:
                raise ImportError(
                    "open3d is required for depth image output. Pin to Python 3.12 and run: uv sync"
                )
            range_depth = max_depth - min_depth
            normalized = (
                (depth_array - min_depth) / range_depth
                if range_depth > 0
                else np.zeros_like(depth_array)
            )
            depth_16bit = (normalized * 65535).astype(np.uint16)
            depth_image = o3d.geometry.Image(depth_16bit)

        point_cloud = None
        if command.return_point_cloud:
            try:
                import open3d as o3d
            except ImportError:
                raise ImportError(
                    "open3d is required for point cloud output. Pin to Python 3.12 and run: uv sync"
                )

            cfg = command.advanced_config
            # Derive intrinsics from image dimensions when not explicitly provided.
            # ~63° horizontal FOV (0.85×width) is a reasonable heuristic for photos.
            fx = cfg.fx if cfg.fx is not None else image.width * 0.85
            fy = cfg.fy if cfg.fy is not None else image.width * 0.85
            cx = cfg.cx if cfg.cx is not None else image.width / 2.0
            cy = cfg.cy if cfg.cy is not None else image.height / 2.0
            color_o3d = o3d.geometry.Image(np.asarray(image).copy())
            depth_o3d = o3d.geometry.Image(
                self._depth_array_to_rgbd_depth(
                    depth_array, _DEPTH_SCALE, _DEPTH_TRUNC, _MIN_DEPTH_M
                )
            )
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=_DEPTH_SCALE,
                depth_trunc=_DEPTH_TRUNC,
                convert_rgb_to_intensity=False,
            )
            generated_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(image.width, image.height, fx, fy, cx, cy),
            )
            point_cloud = generated_point_cloud

        return DepthEstimationResult(
            depth_map=depth_map,
            min_depth=min_depth,
            max_depth=max_depth,
            backend_used=model_id,
            depth_image=depth_image,
            raw_depth=raw_depth,
            point_cloud=point_cloud,
            point_cloud_scale=1.0,
        )

    @staticmethod
    def _depth_array_to_rgbd_depth(
        depth_array: np.ndarray,
        depth_scale: float,
        depth_trunc: float,
        min_depth_m: float = 0.0,
    ) -> np.ndarray:
        min_depth = float(np.nanmin(depth_array))
        max_depth = float(np.nanmax(depth_array))
        depth_range = max_depth - min_depth
        if depth_range <= 0:
            return np.zeros_like(depth_array, dtype=np.uint16)

        # Depth Anything V2 is inverse depth (high value = close, low value = far).
        # Flip so far pixels → high uint16 (large metric depth), close pixels → low.
        # Map onto [min_depth_m, depth_trunc] so close pixels never collapse to Z≈0
        # (which would zero out their X,Y coordinates and destroy foreground geometry).
        # Clip minimum to 1: Open3D silently discards pixels where depth == 0.
        normalized = 1.0 - (depth_array - min_depth) / depth_range
        depth_m = min_depth_m + normalized * (depth_trunc - min_depth_m)
        scaled_depth = depth_m * depth_scale
        return np.clip(scaled_depth, 1, np.iinfo(np.uint16).max).astype(np.uint16)

    @classmethod
    def preload(cls, model_path: str) -> None:
        """Resolve *model_path* (downloading if a URL) and load it into the class-level cache.

        Call this at server startup to ensure the model is in memory before the first request.
        """
        from .defaults import resolve_model_backend

        resolved = resolve_model_backend(model_path)
        cls()._load_depth_anything_checkpoint(resolved)

    @staticmethod
    def _torch_device(torch_module) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_depth_anything_checkpoint(self, model_path: str):
        if model_path in self._depth_anything_models:
            return self._depth_anything_models[model_path]

        with self._model_lock:
            if model_path in self._depth_anything_models:
                return self._depth_anything_models[model_path]

            try:
                import torch
                from transformers import DepthAnythingForDepthEstimation, DPTImageProcessor
            except ImportError as exc:
                raise ImportError(
                    "Depth Anything V2 checkpoints require torch and transformers. Run: uv sync"
                ) from exc

            model = DepthAnythingForDepthEstimation(depth_anything_v2_config(model_path))
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location="cpu")

            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            state_dict = convert_depth_anything_v2_state_dict(state_dict)
            model.load_state_dict(state_dict)
            device = self._torch_device(torch)
            model = model.to(device).eval()

            if device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            processor = DPTImageProcessor(
                size={"height": 518, "width": 518},
                do_resize=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                do_rescale=True,
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )

            self._depth_anything_models[model_path] = (model, processor, torch, device)
            return self._depth_anything_models[model_path]

    def _run_depth_anything_checkpoint(self, model_path: str, image: Image.Image) -> np.ndarray:
        model, processor, torch, device = self._load_depth_anything_checkpoint(model_path)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {name: value.to(device, non_blocking=True) for name, value in inputs.items()}

        device_type = device if isinstance(device, str) else device.type
        if device_type == "cuda":
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        else:
            # MPS float16 can produce degraded depth maps due to limited precision
            # in transformer attention operations.  CPU has no autocast benefit.
            autocast_ctx = contextlib.nullcontext()

        with torch.inference_mode(), autocast_ctx:
            outputs = model(**inputs)

        post_processed = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        depth = post_processed[0]["predicted_depth"]
        result = depth.detach().cpu().numpy().astype(np.float32)

        if device_type == "mps":
            torch.mps.empty_cache()

        return result
