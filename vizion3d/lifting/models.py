import numpy as np
from open3d.geometry import Image as O3dImage  # type: ignore[import-untyped]
from open3d.geometry import PointCloud as O3dPointCloud  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict


class DepthEstimationAdvanceConfig(BaseModel):
    """
    Camera intrinsics for depth estimation.

    All fields default to ``None``, which causes the handler to auto-derive them
    from the input image dimensions (fx = fy ≈ 0.85 × width for ~63° FOV;
    cx/cy at image centre). Supply explicit values for real calibrated cameras.

    Attributes:
        fx: Horizontal focal length in pixels. ``None`` = auto-derived from image
            width. A larger value means a narrower FOV and more perspective compression.
        fy: Vertical focal length in pixels. ``None`` = auto-derived (same as fx).
            Usually equal to ``fx`` for square pixels.
        cx: Principal point x — the pixel column of the optical axis. ``None`` =
            image width / 2.
        cy: Principal point y — the pixel row of the optical axis. ``None`` =
            image height / 2.
    """

    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None


class DepthEstimationResult(BaseModel):
    """
    Result payload returned after a depth estimation inference task.

    Attributes:
        depth_map: Raw floating-point depth array, shape `[H][W]`. Values are
            relative (not metric) for monocular models — closer objects have
            higher values for inverse-depth outputs.
        min_depth: Minimum value in `depth_map`.
        max_depth: Maximum value in `depth_map`. Guaranteed `max_depth >= min_depth`.
        backend_used: Resolved model identifier that processed the request
            (local file path).
        depth_image: 16-bit grayscale `open3d.geometry.Image` (dtype `uint16`),
            present by default (suppress with `return_depth_image=False`).
            Depth Anything V2 outputs inverse relative depth, so higher uint16 values
            correspond to closer pixels — closer = brighter.
        raw_depth: Raw float32 depth array, shape `(H, W)`, present by default
            (suppress with `return_raw_depth=False`).  Values are relative —
            not metric — for monocular depth estimation.
        point_cloud: Coloured `open3d.geometry.PointCloud` unprojected from the
            RGB-D image, present when `return_point_cloud=True`. Coordinates use
            the OpenGL/viewer convention (X+ right, Y+ up, Z- forward) and are in
            metres — multiply distances by `point_cloud_scale` (always `1.0`) to
            confirm the unit.
        point_cloud_scale: Scale factor for the point cloud coordinate space.
            Multiply any distance measured between two points in the returned
            point cloud by this value to get the equivalent distance in metres.
            Always `1.0` — Open3D produces point cloud coordinates directly in metres.
    """

    depth_map: list[list[float]]
    min_depth: float
    max_depth: float
    backend_used: str
    depth_image: O3dImage | None = None
    raw_depth: np.ndarray | None = None
    point_cloud: O3dPointCloud | None = None
    point_cloud_scale: float = 1.0

    model_config = ConfigDict(arbitrary_types_allowed=True)
