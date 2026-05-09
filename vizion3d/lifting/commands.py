from dataclasses import dataclass, field

from vizion3d.core.cqrs import Command

from .defaults import DEFAULT_DEPTH_MODEL_URL
from .models import DepthEstimationAdvanceConfig, DepthEstimationResult


@dataclass
class DepthEstimationCommand(Command[DepthEstimationResult]):
    """
    Command payload to trigger a depth estimation inference task.

    Attributes:
        image_input: The input image. Pass a file-path string or raw image bytes.
            The handler auto-detects which form is supplied.
        model_backend: Model backend to use for inference.

            - Default value is the vizion3D release checkpoint URL
              (`depth_anything_v2_vitb.pth`), which is downloaded on first use and
              cached under `~/.cache/vizion3d/models/`.
              Set `VIZION3D_MODEL_CACHE` to override the cache directory.
            - A local `.pth` or `.pt` path is loaded directly as a Depth Anything V2
              checkpoint — no download occurs.
            - Any HTTPS URL is downloaded to the cache directory and loaded as a
              checkpoint.

        return_depth_image: When `True`, the result includes a 16-bit grayscale
            `open3d.geometry.Image` (dtype `uint16`).  Depth Anything V2 outputs
            inverse relative depth (higher = closer), so higher uint16 values
            correspond to closer pixels — closer = brighter.
            Requires Open3D (Python 3.12).
        return_raw_depth: When `True`, the result includes the raw depth array
            as a float32 numpy array of shape `(H, W)`.  Values are relative
            (not metric) for monocular depth — unmodified output from the model.
        return_point_cloud: When `True`, the result includes an
            `open3d.geometry.PointCloud` unprojected from the RGB-D image using
            the camera intrinsics in `advanced_config`. Point coordinates are in metres.
            Requires Open3D (Python 3.12).
        advanced_config: Camera intrinsics and depth range settings. Override any
            field to customise — e.g.
            ``advanced_config=DepthEstimationAdvanceConfig(fx=615.0, fy=615.0)``.
            Unspecified fields keep their defaults (PrimeSense values).
    """

    image_input: str | bytes
    model_backend: str = DEFAULT_DEPTH_MODEL_URL
    return_depth_image: bool = True
    return_raw_depth: bool = True
    return_point_cloud: bool = False
    advanced_config: DepthEstimationAdvanceConfig = field(
        default_factory=DepthEstimationAdvanceConfig
    )
