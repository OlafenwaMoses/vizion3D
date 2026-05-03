import contextlib
import io
import threading

import numpy as np
from PIL import Image

from vizion3d.core.cqrs import CommandHandler

from .commands import DepthEstimationCommand
from .defaults import resolve_model_backend
from .depth_anything import convert_depth_anything_v2_state_dict, depth_anything_v2_config
from .models import DepthEstimationResult

# Legacy module-level constants kept for reference — actual values come from
# DepthEstimationAdvanceConfig on each command.
_DEFAULT_DEPTH_SCALE = 1000.0
_DEFAULT_DEPTH_TRUNC = 10.0
OPEN3D_CAMERA_TO_IMAGE_VIEW_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


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

        depth_image = None
        if command.return_depth_image:
            try:
                import open3d as o3d
            except ImportError:
                raise ImportError(
                    "open3d is required for depth image output. "
                    "Pin to Python 3.12 and run: uv sync"
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
        mesh = None
        if command.return_point_cloud or command.return_mesh:
            try:
                import open3d as o3d
            except ImportError:
                raise ImportError(
                    "open3d is required for point cloud / mesh output. "
                    "Pin to Python 3.12 and run: uv sync"
                )

            cfg = command.advanced_config
            color_o3d = o3d.geometry.Image(np.asarray(image).copy())
            depth_o3d = o3d.geometry.Image(
                self._depth_array_to_rgbd_depth(depth_array, cfg.depth_scale, cfg.depth_trunc)
            )
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=cfg.depth_scale,
                depth_trunc=cfg.depth_trunc,
                convert_rgb_to_intensity=False,
            )
            generated_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    image.width, image.height, cfg.fx, cfg.fy, cfg.cx, cfg.cy
                ),
            )
            self._orient_point_cloud_like_image(generated_point_cloud)

            if command.return_point_cloud:
                point_cloud = generated_point_cloud

            if command.return_mesh:
                mesh = self._mesh_from_point_cloud(generated_point_cloud, o3d)

        return DepthEstimationResult(
            depth_map=depth_map,
            min_depth=min_depth,
            max_depth=max_depth,
            backend_used=model_id,
            depth_image=depth_image,
            point_cloud=point_cloud,
            mesh=mesh,
            point_cloud_scale=1.0,
        )

    @staticmethod
    def _orient_point_cloud_like_image(point_cloud):
        point_cloud.transform(OPEN3D_CAMERA_TO_IMAGE_VIEW_TRANSFORM)
        return point_cloud

    @staticmethod
    def _depth_array_to_rgbd_depth(
        depth_array: np.ndarray, depth_scale: float, depth_trunc: float
    ) -> np.ndarray:
        min_depth = float(np.nanmin(depth_array))
        max_depth = float(np.nanmax(depth_array))
        depth_range = max_depth - min_depth
        if depth_range <= 0:
            return np.zeros_like(depth_array, dtype=np.uint16)

        normalized = (depth_array - min_depth) / depth_range
        scaled_depth = normalized * depth_trunc * depth_scale
        return np.clip(scaled_depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    @staticmethod
    def _mesh_from_point_cloud(point_cloud, o3d):
        if point_cloud.is_empty():
            return o3d.geometry.TriangleMesh()

        point_cloud = o3d.geometry.PointCloud(point_cloud)
        if not point_cloud.has_normals():
            point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

        distances = point_cloud.compute_nearest_neighbor_distance()
        if len(distances) == 0:
            return o3d.geometry.TriangleMesh()

        radius = float(np.mean(distances) * 3.0)
        if radius <= 0:
            return o3d.geometry.TriangleMesh()

        return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud,
            o3d.utility.DoubleVector([radius, radius * 2.0]),
        )

    @staticmethod
    def _torch_device(torch_module) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
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
                    "Depth Anything V2 checkpoints require torch and transformers. "
                    "Run: uv sync"
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
        if device_type == 'cuda':
            autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
        elif device_type == 'mps':
            autocast_ctx = torch.amp.autocast(device_type='mps', dtype=torch.float16, enabled=True)
        else:
            autocast_ctx = contextlib.nullcontext()

        with torch.inference_mode(), autocast_ctx:
            outputs = model(**inputs)

        post_processed = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        depth = post_processed[0]["predicted_depth"]
        result = depth.detach().cpu().numpy().astype(np.float32)

        if device_type == 'mps':
            torch.mps.empty_cache()

        return result
