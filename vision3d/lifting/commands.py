from dataclasses import dataclass

from vision3d.core.cqrs import Command

from .defaults import DEFAULT_DEPTH_MODEL_BACKEND
from .models import DepthEstimationResult


@dataclass
class DepthEstimationCommand(Command[DepthEstimationResult]):
    """
    Command payload to trigger a depth estimation inference task.

    Attributes:
        image_input: The input image. Pass a file-path string or raw image bytes.
            The handler auto-detects which form is supplied.
        model_backend: Model backend to use for inference.

            - Default value (`"depth-anything/Depth-Anything-V2-Base-hf"`) resolves to
              the vision3D release checkpoint (`depth_anything_v2_vitb.pth`), which is
              downloaded on first use and cached under `~/.cache/vision3d/models/`.
              Set `VISION3D_MODEL_CACHE` to override the cache directory.
            - A local `.pth` or `.pt` path is loaded directly as a Depth Anything V2
              checkpoint — no download occurs.
            - Any other string is forwarded to
              `transformers.pipeline(task="depth-estimation", model=...)`.

        return_depth_image: When `True`, the result includes a 16-bit grayscale
            `open3d.geometry.Image` (dtype `uint16`) mapping `[min_depth, max_depth]`
            to the full 0–65535 range. Requires Open3D (Python 3.12).
        return_point_cloud: When `True`, the result includes an
            `open3d.geometry.PointCloud` unprojected from the RGB-D image using
            PrimeSense default camera intrinsics. Point coordinates are in metres.
            Requires Open3D (Python 3.12).
        return_mesh: When `True`, the result includes an
            `open3d.geometry.TriangleMesh` reconstructed from the point cloud via
            ball-pivoting. Includes vertex colours. Requires Open3D (Python 3.12).
    """

    image_input: str | bytes
    model_backend: str = DEFAULT_DEPTH_MODEL_BACKEND
    return_depth_image: bool = False
    return_point_cloud: bool = False
    return_mesh: bool = False
