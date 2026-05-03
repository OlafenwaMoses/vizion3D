"""
CQRS command payload for the Stereo Depth task.
"""

from dataclasses import dataclass, field

from vizion3d.core.cqrs import Command

from .defaults import DEFAULT_STEREO_MODEL_URL
from .models import StereoDepthAdvancedConfig, StereoDepthResult


@dataclass
class StereoDepthCommand(Command[StereoDepthResult]):
    """Command payload to trigger a stereo depth inference task.

    Stereo depth produces **real metric depth** (in metres) by matching
    corresponding pixels across a rectified left/right image pair and applying
    the stereo geometry formula:

        depth_m = baseline_mm × focal_length_px / disparity_px / 1000

    Attributes:
        left_image: The left-camera image.  Pass a file-path string or raw
            image bytes.  The handler auto-detects which form is supplied.
        right_image: The right-camera image (same resolution as *left_image*,
            from a horizontally-offset camera).  Same path/bytes convention.
        model_backend: S2M2 checkpoint to use for inference.

            - Default value is the vizion3D release checkpoint URL
              (``stereo-depth-s2m2-L.pth``), downloaded on first use and cached
              under ``~/.cache/vizion3d/models/``.
              Set ``VIZION3D_MODEL_CACHE`` to override the cache directory.
            - A local ``.pth`` or ``.pt`` path is loaded directly.
            - Any HTTPS URL is downloaded to the cache directory and loaded.

        return_depth_image: When ``True``, the result includes a 16-bit grayscale
            ``open3d.geometry.Image`` (dtype ``uint16``) where the full 0–65535
            range maps linearly to ``[min_depth, max_depth]`` in metres.
        return_point_cloud: When ``True``, the result includes an
            ``open3d.geometry.PointCloud`` unprojected using the stereo camera
            intrinsics in ``advanced_config``. Point coordinates are in metres.
        return_mesh: When ``True``, the result includes an
            ``open3d.geometry.TriangleMesh`` reconstructed from the point cloud
            via ball-pivoting. Includes vertex colours.
        advanced_config: Camera intrinsics and inference settings. Override any
            field to match your stereo rig — e.g.
            ``advanced_config=StereoDepthAdvancedConfig(focal_length=1733.74,
            cx=792.27, cy=541.89, baseline=536.62)``.
            Unspecified fields keep their defaults (1280×720 @ 100 mm baseline).
    """

    left_image: str | bytes
    right_image: str | bytes
    model_backend: str = DEFAULT_STEREO_MODEL_URL
    return_depth_image: bool = False
    return_point_cloud: bool = False
    return_mesh: bool = False
    advanced_config: StereoDepthAdvancedConfig = field(default_factory=StereoDepthAdvancedConfig)
