"""
Data models for the Stereo Depth task.

Defines the camera-configuration Pydantic model and the result payload.
"""

from open3d.geometry import Image as O3dImage  # type: ignore[import-untyped]
from open3d.geometry import PointCloud as O3dPointCloud  # type: ignore[import-untyped]
from open3d.geometry import TriangleMesh as O3dTriangleMesh  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict


class StereoDepthAdvancedConfig(BaseModel):
    """Camera intrinsics and inference settings for stereo depth estimation.

    All fields are optional overrides — unspecified fields retain sensible defaults
    for a 1280×720 commodity stereo camera with a 100 mm baseline.

    Attributes:
        focal_length: Focal length in pixels (assumes square pixels, i.e. fx = fy).
            Larger values mean a narrower field of view and more perspective
            compression. Override with your camera's actual calibrated value.
        cx: Principal point x — the pixel column of the optical axis, typically
            near the horizontal image centre (``image_width / 2 - 0.5``).
        cy: Principal point y — the pixel row of the optical axis, typically near
            the vertical image centre (``image_height / 2 - 0.5``).
        baseline: Stereo baseline in **millimetres** — the physical distance between
            the two camera optical centres. Real metric depth is proportional to
            this value, so an incorrect baseline scales all depth values uniformly.
        doffs: Disparity offset in pixels. Non-zero for Middlebury-style calibration
            where the principal points are not aligned across the two views.
            Set to ``0.0`` for standard rectified pairs.
        z_far: Maximum depth in metres. Points beyond this distance are excluded
            from the point cloud and mesh to reduce noise and file size.
        conf_threshold: Minimum per-pixel confidence score (in ``[0, 1]``) for a
            point to be included in the point cloud.  Lower values include more
            uncertain points; higher values give sparser but more reliable clouds.
        occ_threshold: Minimum occlusion score (in ``[0, 1]``) for a point to be
            included. Points with low occlusion scores are likely partially occluded
            in one view and produce less reliable depth estimates.
        scale_factor: Input image downscale factor before inference.  ``1.0`` means
            full resolution (highest quality, slowest).  ``0.5`` halves both spatial
            dimensions (~3–4× faster at some quality cost).
    """

    focal_length: float = 1000.0
    cx: float = 640.0
    cy: float = 360.0
    baseline: float = 100.0
    doffs: float = 0.0
    z_far: float = 10.0
    conf_threshold: float = 0.1
    occ_threshold: float = 0.5
    scale_factor: float = 1.0


class StereoDepthResult(BaseModel):
    """Result payload returned after a stereo depth inference task.

    Attributes:
        depth_map: Metric depth in **metres**, shape ``[H][W]``.  Unlike monocular
            depth estimation, these are real-world distances (assuming correct camera
            calibration) — not relative or fictitious values.
        disparity_map: Raw disparity map in pixels, shape ``[H][W]``.  Disparity
            is the horizontal pixel offset between matched features across the
            left and right images. Depth = baseline × focal_length / disparity.
        min_depth: Minimum value in ``depth_map`` (metres).
        max_depth: Maximum value in ``depth_map`` (metres). Guaranteed
            ``max_depth >= min_depth``.
        backend_used: Resolved local file path of the checkpoint used.
        depth_image: 16-bit grayscale ``open3d.geometry.Image`` (dtype ``uint16``)
            where the full 0–65535 range maps linearly to ``[min_depth, max_depth]``.
            Present when ``return_depth_image=True`` was set on the command.
        point_cloud: Coloured ``open3d.geometry.PointCloud`` unprojected from the
            RGB-D image using the camera intrinsics in ``advanced_config``.
            Coordinates are in metres.  Present when ``return_point_cloud=True``.
        mesh: ``open3d.geometry.TriangleMesh`` reconstructed via ball-pivoting
            from the point cloud.  Includes vertex colours.
            Present when ``return_mesh=True``.
        point_cloud_scale: Scale factor: multiply any distance measured between
            two points in the returned point cloud by this value to get the
            equivalent distance in metres.  Always ``1.0`` for stereo depth —
            coordinates are already in real metric units.
    """

    depth_map: list[list[float]]
    disparity_map: list[list[float]]
    min_depth: float
    max_depth: float
    backend_used: str
    depth_image: O3dImage | None = None
    point_cloud: O3dPointCloud | None = None
    mesh: O3dTriangleMesh | None = None
    point_cloud_scale: float = 1.0

    model_config = ConfigDict(arbitrary_types_allowed=True)
