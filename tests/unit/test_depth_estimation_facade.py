import pytest
from PIL import Image

from vizion3d.lifting import DepthEstimation, DepthEstimationCommand
from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_FILENAME

o3d = pytest.importorskip(
    "open3d",
    reason="open3d requires Python 3.12 - run: uv python pin 3.12 && uv sync",
)


def test_depth_estimation_basic(dummy_image_bytes):
    res = DepthEstimation().run(DepthEstimationCommand(image_input=dummy_image_bytes))

    assert isinstance(res.depth_map, list)
    assert len(res.depth_map) > 0
    assert isinstance(res.depth_map[0], list)
    assert res.max_depth >= res.min_depth
    assert res.backend_used.endswith(DEFAULT_DEPTH_MODEL_FILENAME)
    assert isinstance(res.depth_image, o3d.geometry.Image), "depth_image must be present by default"
    assert res.raw_depth is not None, "raw_depth must be present by default"
    assert res.point_cloud is None
    assert res.point_cloud_scale == 1.0


def test_depth_estimation_suppresses_depth_image_and_raw_depth(dummy_image_bytes):
    res = DepthEstimation().run(
        DepthEstimationCommand(
            image_input=dummy_image_bytes,
            return_depth_image=False,
            return_raw_depth=False,
        )
    )
    assert res.depth_image is None
    assert res.raw_depth is None


def test_depth_estimation_returns_depth_image(dummy_image_bytes):
    import numpy as np

    res = DepthEstimation().run(
        DepthEstimationCommand(image_input=dummy_image_bytes, return_depth_image=True)
    )

    assert isinstance(res.depth_image, o3d.geometry.Image)
    arr = np.asarray(res.depth_image)
    assert arr.dtype == np.uint16
    assert arr.shape == (100, 100)


def test_depth_estimation_returns_raw_depth(dummy_image_bytes):
    import numpy as np

    res = DepthEstimation().run(
        DepthEstimationCommand(image_input=dummy_image_bytes, return_raw_depth=True)
    )

    assert res.raw_depth is not None
    assert res.raw_depth.dtype == np.float32
    assert res.raw_depth.shape == (100, 100)


def test_raw_depth_matches_depth_map(dummy_image_bytes):
    import numpy as np

    res = DepthEstimation().run(
        DepthEstimationCommand(image_input=dummy_image_bytes, return_raw_depth=True)
    )

    depth_map_arr = np.array(res.depth_map, dtype=np.float32)
    np.testing.assert_array_equal(res.raw_depth, depth_map_arr)


def test_depth_estimation_returns_point_cloud(dummy_image_bytes):
    res = DepthEstimation().run(
        DepthEstimationCommand(image_input=dummy_image_bytes, return_point_cloud=True)
    )

    assert isinstance(res.point_cloud, o3d.geometry.PointCloud)
    assert res.point_cloud.has_points()
    assert res.point_cloud.has_colors()
    import numpy as np

    points = np.asarray(res.point_cloud.points)
    colors = np.asarray(res.point_cloud.colors)
    assert 0 < len(points) <= 100 * 100
    assert len(colors) == len(points)
    assert res.point_cloud_scale == 1.0


def test_point_cloud_scale_is_metres_per_unit(dummy_image_bytes):
    res = DepthEstimation().run(
        DepthEstimationCommand(image_input=dummy_image_bytes, return_point_cloud=True)
    )

    assert isinstance(res.point_cloud_scale, float)
    assert res.point_cloud_scale == 1.0

    import numpy as np

    points = np.asarray(res.point_cloud.points)
    p0, p1 = points[0], points[1]
    dist_units = float(np.linalg.norm(p0 - p1))
    dist_metres = dist_units * res.point_cloud_scale
    assert dist_metres == pytest.approx(dist_units)


def test_point_cloud_is_in_opengl_camera_space(dummy_image_bytes):
    """Point cloud must be in OpenGL/viewer camera-space coordinates:
    X+ right, Y+ up, Z- forward (all valid scene Z negative).
    """
    import numpy as np

    res = DepthEstimation().run(
        DepthEstimationCommand(image_input=dummy_image_bytes, return_point_cloud=True)
    )
    points = np.asarray(res.point_cloud.points)

    assert points.shape[1] == 3
    assert np.all(points[:, 2] < 0), "Z must be negative (forward from camera in OpenGL space)"


def test_depth_estimation_accepts_file_path(tmp_path):
    img = Image.new("RGB", (50, 50), color="red")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    res = DepthEstimation().run(DepthEstimationCommand(image_input=str(img_path)))

    assert isinstance(res.depth_map, list)
    assert res.max_depth >= res.min_depth
