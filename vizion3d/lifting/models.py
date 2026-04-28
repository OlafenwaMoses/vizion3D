from pydantic import BaseModel, ConfigDict

from open3d.geometry import Image as O3dImage  # type: ignore[import-untyped]
from open3d.geometry import PointCloud as O3dPointCloud  # type: ignore[import-untyped]
from open3d.geometry import TriangleMesh as O3dTriangleMesh  # type: ignore[import-untyped]


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
            (local file path or Hugging Face model ID).
        depth_image: 16-bit grayscale `open3d.geometry.Image` (dtype `uint16`),
            present when `return_depth_image=True` was set on the command.
            The full 0–65535 range maps linearly to `[min_depth, max_depth]`.
        point_cloud: Coloured `open3d.geometry.PointCloud` unprojected from the
            RGB-D image, present when `return_point_cloud=True`. Coordinates are
            in metres — multiply distances by `point_cloud_scale` (always `1.0`)
            to confirm the unit.
        mesh: `open3d.geometry.TriangleMesh` reconstructed via ball-pivoting,
            present when `return_mesh=True`. Includes vertex colours.
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
    point_cloud: O3dPointCloud | None = None
    mesh: O3dTriangleMesh | None = None
    point_cloud_scale: float = 1.0

    model_config = ConfigDict(arbitrary_types_allowed=True)
