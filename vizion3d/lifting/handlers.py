import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from vizion3d.core.cqrs import CommandHandler

from .commands import DepthEstimationCommand
from .defaults import resolve_model_backend
from .depth_anything import convert_depth_anything_v2_state_dict, depth_anything_v2_config
from .models import DepthEstimationResult

RGBD_DEPTH_SCALE = 1000.0   # default: uint16 millimetres (RealSense / Kinect / PrimeSense)
RGBD_DEPTH_TRUNC = 10.0    # default: discard points beyond 10 m
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
    _pipelines = {}
    _depth_anything_models = {}
    _model_lock = threading.Lock()

    def handle(self, command: DepthEstimationCommand) -> DepthEstimationResult:
        model_id = resolve_model_backend(command.model_backend)

        if isinstance(command.image_input, str):
            image = Image.open(command.image_input).convert("RGB")
        else:
            image = Image.open(io.BytesIO(command.image_input)).convert("RGB")

        if self._is_depth_anything_checkpoint(model_id):
            depth_array = self._run_depth_anything_checkpoint(model_id, image)
        else:
            result = self._load_hugging_face_pipeline(model_id)(image)
            depth_tensor = result["predicted_depth"]
            depth_array = depth_tensor.squeeze().cpu().numpy().astype(float)

        min_depth = float(np.min(depth_array))
        max_depth = float(np.max(depth_array))
        depth_map = depth_array.tolist()

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

            color_o3d = o3d.geometry.Image(np.asarray(image).copy())
            depth_o3d = o3d.geometry.Image(
                self._depth_array_to_rgbd_depth(depth_array)
            )
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=RGBD_DEPTH_SCALE,
                depth_trunc=RGBD_DEPTH_TRUNC,
                convert_rgb_to_intensity=False,
            )
            generated_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
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
    def _is_depth_anything_checkpoint(model_id: str) -> bool:
        return Path(model_id).suffix.lower() in {".pth", ".pt"}

    @staticmethod
    def _orient_point_cloud_like_image(point_cloud):
        point_cloud.transform(OPEN3D_CAMERA_TO_IMAGE_VIEW_TRANSFORM)
        return point_cloud

    @staticmethod
    def _depth_array_to_rgbd_depth(depth_array: np.ndarray) -> np.ndarray:
        min_depth = float(np.nanmin(depth_array))
        max_depth = float(np.nanmax(depth_array))
        depth_range = max_depth - min_depth
        if depth_range <= 0:
            return np.zeros_like(depth_array, dtype=np.uint16)

        normalized = (depth_array - min_depth) / depth_range
        scaled_depth = normalized * RGBD_DEPTH_TRUNC * RGBD_DEPTH_SCALE
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

    def _load_hugging_face_pipeline(self, model_id: str):
        if model_id in self._pipelines:
            return self._pipelines[model_id]

        with self._model_lock:
            if model_id in self._pipelines:
                return self._pipelines[model_id]

            from transformers import pipeline

            self._pipelines[model_id] = pipeline(
                task="depth-estimation", model=model_id, trust_remote_code=True
            )
            return self._pipelines[model_id]

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

            self._depth_anything_models[model_path] = (model, processor, torch)
            return self._depth_anything_models[model_path]

    def _run_depth_anything_checkpoint(self, model_path: str, image: Image.Image) -> np.ndarray:
        model, processor, torch = self._load_depth_anything_checkpoint(model_path)
        device = next(model.parameters()).device
        inputs = processor(images=image, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        post_processed = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        depth = post_processed[0]["predicted_depth"]

        return depth.detach().cpu().numpy().astype(float)
