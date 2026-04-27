import pytest
from PIL import Image

from vision3d.lifting import DepthEstimation, DepthEstimationCommand
from vision3d.lifting.defaults import DEFAULT_DEPTH_MODEL_FILENAME
from vision3d.lifting.handlers import DepthEstimationHandler
from vision3d.reconstruction import SfMCommand, StructureFromMotion

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
    assert res.depth_image is None
    assert res.point_cloud is None
    assert res.mesh is None
    assert res.point_cloud_scale == 1.0


def test_depth_estimation_returns_depth_image(dummy_image_bytes):
    res = DepthEstimation().run(
        DepthEstimationCommand(image_input=dummy_image_bytes, return_depth_image=True)
    )

    assert isinstance(res.depth_image, o3d.geometry.Image)
    import numpy as np
    arr = np.asarray(res.depth_image)
    assert arr.dtype == np.uint16
    assert arr.shape == (100, 100)


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


def test_point_cloud_orientation_keeps_image_top_up():
    import numpy as np

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array(
            [
                [0.0, -1.0, 2.0],  # image top in Open3D camera coordinates
                [0.0, 1.0, 2.0],  # image bottom in Open3D camera coordinates
            ]
        )
    )

    DepthEstimationHandler._orient_point_cloud_like_image(pcd)
    points = np.asarray(pcd.points)

    assert points[0, 1] > points[1, 1]
    assert np.all(points[:, 2] > 0)


def test_depth_estimation_returns_mesh(dummy_image_bytes):
    res = DepthEstimation().run(
        DepthEstimationCommand(image_input=dummy_image_bytes, return_mesh=True)
    )

    assert isinstance(res.mesh, o3d.geometry.TriangleMesh)
    assert res.mesh.has_vertices()
    assert res.mesh.has_triangles()
    assert res.mesh.has_vertex_colors()
    import numpy as np

    assert len(np.asarray(res.mesh.triangles)) > 0


def test_depth_estimation_accepts_file_path(tmp_path):
    img = Image.new("RGB", (50, 50), color="red")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    res = DepthEstimation().run(DepthEstimationCommand(image_input=str(img_path)))

    assert isinstance(res.depth_map, list)
    assert res.max_depth >= res.min_depth


def test_sfm_returns_correct_mock_data():
    res = StructureFromMotion().run(SfMCommand(images={"img1": b"fake", "img2": b"fake2"}))

    assert len(res.sparse_point_cloud) == 2
    assert len(res.camera_poses) == 2
    assert res.backend_used == "colmap-mock"
    assert {p.image_id for p in res.camera_poses} == {"img1", "img2"}

    for pose in res.camera_poses:
        assert len(pose.rotation_matrix) == 3
        assert len(pose.rotation_matrix[0]) == 3
        assert len(pose.translation_vector) == 3


def test_sfm_single_image():
    res = StructureFromMotion().run(SfMCommand(images={"only_img": b"data"}))

    assert len(res.camera_poses) == 1
    assert res.camera_poses[0].image_id == "only_img"
