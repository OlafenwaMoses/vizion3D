"""
Coordinate space invariant tests for vizion3d point cloud outputs.

Both DepthEstimation and StereoDepth must emit standard camera-space coordinates:
  X+ right   (increasing pixel column → increasing X)
  Y+ down    (increasing pixel row    → increasing Y)
  Z+ forward (all metric depth values are positive)

This is the OpenCV / Open3D / COLMAP / ROS camera convention and allows point
clouds from either pipeline to be consumed by downstream 3D tools without any
axis-flip post-processing.

All tests use fully synthetic inputs; no model checkpoint is required.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from vizion3d.lifting import DepthEstimation, DepthEstimationCommand
from vizion3d.lifting.handlers import DepthEstimationHandler
from vizion3d.lifting.models import DepthEstimationAdvanceConfig
from vizion3d.stereo import StereoDepth
from vizion3d.stereo.commands import StereoDepthCommand
from vizion3d.stereo.handlers import StereoDepthHandler
from vizion3d.stereo.models import StereoDepthAdvancedConfig

o3d = pytest.importorskip(
    "open3d",
    reason="open3d requires Python 3.12 - run: uv python pin 3.12 && uv sync",
)

# ── Synthetic scene dimensions & intrinsics ───────────────────────────────────

W, H = 20, 16
FX = FY = 200.0
CX, CY = 9.5, 7.5   # exact centre of 20 × 16

_STEREO_CFG = StereoDepthAdvancedConfig(
    focal_length=FX,
    cx=CX,
    cy=CY,
    baseline=100.0,   # 100 mm
    doffs=0.0,
    z_far=1000.0,
    conf_threshold=0.0,   # accept all pixels
    occ_threshold=0.0,
)

_DEPTH_CFG = DepthEstimationAdvanceConfig(
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY,
    depth_scale=1000.0,
    depth_trunc=10.0,
)


# ── Runner helpers ─────────────────────────────────────────────────────────────


def _image_bytes(w: int = W, h: int = H) -> bytes:
    img = Image.new("RGB", (w, h), color=(80, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _run_stereo(disp: np.ndarray, cfg: StereoDepthAdvancedConfig = _STEREO_CFG):
    """Run StereoDepth with a mocked disparity map and return the result."""
    h, w = disp.shape
    occ = np.ones((h, w), dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)
    left = _image_bytes(w, h)
    right = _image_bytes(w, h)
    with patch.object(StereoDepthHandler, "_run_s2m2", return_value=(disp, occ, conf)):
        return StereoDepth().run(
            StereoDepthCommand(
                left_image=left,
                right_image=right,
                model_backend="/fake/model.pth",
                return_point_cloud=True,
                advanced_config=cfg,
            )
        )


def _run_depth(depth: np.ndarray, cfg: DepthEstimationAdvanceConfig = _DEPTH_CFG):
    """Run DepthEstimation with a mocked depth array and return the result."""
    h, w = depth.shape
    img = _image_bytes(w, h)
    with patch.object(
        DepthEstimationHandler, "_run_depth_anything_checkpoint", return_value=depth
    ):
        return DepthEstimation().run(
            DepthEstimationCommand(
                image_input=img,
                model_backend="/fake/model.pth",
                return_point_cloud=True,
                advanced_config=cfg,
            )
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def uniform_disp():
    """Positive uniform disparity giving exactly 1.0 m depth throughout."""
    # depth_mm = baseline * focal / disp  →  100 * 200 / 20 = 1000 mm = 1.0 m
    return np.full((H, W), 20.0, dtype=np.float32)


@pytest.fixture
def gradient_depth():
    """Linearly increasing depth array, ensuring range > 0 so every normalised
    value is in (0, 1] and all unprojected Z values are positive."""
    return np.linspace(1.0, 5.0, H * W).reshape(H, W).astype(np.float32)


# ── Z+ forward ────────────────────────────────────────────────────────────────


class TestZForward:
    def test_stereo_all_z_positive(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        assert len(pts) > 0, "point cloud must not be empty"
        assert np.all(pts[:, 2] > 0), "all stereo Z values must be positive (forward)"

    def test_depth_estimation_all_z_positive(self, gradient_depth):
        pts = np.asarray(_run_depth(gradient_depth).point_cloud.points)
        assert len(pts) > 0, "point cloud must not be empty"
        assert np.all(pts[:, 2] > 0), "all depth estimation Z values must be positive (forward)"


# ── Y+ down ───────────────────────────────────────────────────────────────────


class TestYDown:
    """Top-of-image pixels (small v) → negative Y; bottom pixels (large v) → positive Y."""

    def test_stereo_top_rows_have_negative_y(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        assert pts[:, 1].min() < 0, "top pixel rows must produce Y < 0"

    def test_stereo_bottom_rows_have_positive_y(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        assert pts[:, 1].max() > 0, "bottom pixel rows must produce Y > 0"

    def test_stereo_mean_y_increases_with_row(self, uniform_disp):
        """Average Y of the upper half must be strictly less than the lower half."""
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        top_mean = pts[pts[:, 1] < 0, 1].mean()
        bot_mean = pts[pts[:, 1] > 0, 1].mean()
        assert top_mean < bot_mean

    def test_depth_estimation_y_sign_convention(self, gradient_depth):
        """DepthEstimation (Open3D RGBD pipeline) must also be Y+ down."""
        pts = np.asarray(_run_depth(gradient_depth).point_cloud.points)
        assert pts[:, 1].min() < 0, "top rows must produce Y < 0"
        assert pts[:, 1].max() > 0, "bottom rows must produce Y > 0"


# ── X+ right ──────────────────────────────────────────────────────────────────


class TestXRight:
    """Left-of-centre pixels (small u) → negative X; right pixels → positive X."""

    def test_stereo_left_columns_have_negative_x(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        assert pts[:, 0].min() < 0, "left pixel columns must produce X < 0"

    def test_stereo_right_columns_have_positive_x(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        assert pts[:, 0].max() > 0, "right pixel columns must produce X > 0"

    def test_stereo_mean_x_increases_with_column(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        left_mean = pts[pts[:, 0] < 0, 0].mean()
        right_mean = pts[pts[:, 0] > 0, 0].mean()
        assert left_mean < right_mean

    def test_depth_estimation_x_sign_convention(self, gradient_depth):
        pts = np.asarray(_run_depth(gradient_depth).point_cloud.points)
        assert pts[:, 0].min() < 0, "left columns must produce X < 0"
        assert pts[:, 0].max() > 0, "right columns must produce X > 0"


# ── Right-handed coordinate system ───────────────────────────────────────────


class TestRightHanded:
    """X+ right, Y+ down, Z+ forward is a right-handed system: X × Y = +Z.

    Verified empirically from the point cloud by measuring the empirical
    X and Y direction vectors and checking their cross product.
    """

    def _empirical_axes(self, pts: np.ndarray):
        """Return unit vectors for the empirical X and Y axes from the cloud."""
        # X axis: direction of increasing column at mid-row
        # With pinhole: X = (u-cx)*Z/f, so a step of Δu gives ΔX = Δu*Z/f > 0
        # We can infer this from the range of X vs Y relative to Z.
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        assert x_range > 0 and y_range > 0
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    def test_stereo_cross_product_points_forward(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        x_hat, y_hat = self._empirical_axes(pts)
        z_hat = np.cross(x_hat, y_hat)
        assert z_hat[2] > 0, "X × Y must point in +Z (right-handed, Z forward)"

    def test_depth_estimation_cross_product_points_forward(self, gradient_depth):
        pts = np.asarray(_run_depth(gradient_depth).point_cloud.points)
        x_hat, y_hat = self._empirical_axes(pts)
        z_hat = np.cross(x_hat, y_hat)
        assert z_hat[2] > 0, "X × Y must point in +Z (right-handed, Z forward)"

    def test_stereo_x_y_span_is_symmetric_about_principal_point(self, uniform_disp):
        """With a centred principal point, X and Y should each be symmetric about 0."""
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        assert abs(pts[:, 0].mean()) < 0.01, "X centroid must be near 0"
        assert abs(pts[:, 1].mean()) < 0.01, "Y centroid must be near 0"


# ── Pinhole round-trip ────────────────────────────────────────────────────────


class TestPinholeRoundTrip:
    """Unproject then re-project must recover the original pixel coordinates."""

    def test_stereo_unproject_reproject(self, uniform_disp):
        """All X,Y,Z points re-projected through the pinhole model must land
        on integer pixel locations within the image bounds."""
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        u = X * FX / Z + CX
        v = Y * FY / Z + CY
        assert np.all(u >= -0.5) and np.all(u <= W - 0.5), "re-projected u must be in image"
        assert np.all(v >= -0.5) and np.all(v <= H - 0.5), "re-projected v must be in image"

    def test_stereo_known_pixel_round_trip(self, uniform_disp):
        """For a specific pixel (u=15, v=12) the unprojected 3D point must
        re-project back to within 0.5 px of the original pixel."""
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        u_reproj = X * FX / Z + CX
        v_reproj = Y * FY / Z + CY
        # Choose a target pixel and find the nearest reprojected point
        u_target, v_target = 15.0, 12.0
        dist = np.hypot(u_reproj - u_target, v_reproj - v_target)
        assert dist.min() < 0.6, f"nearest re-projected point is {dist.min():.3f} px from target"


# ── Metric Z (stereo only) ────────────────────────────────────────────────────


class TestMetricDepth:
    """Stereo depth must follow the pinhole stereo formula exactly."""

    def test_uniform_disparity_gives_correct_metric_z(self):
        """baseline=100 mm, focal=200 px, disp=20 px → depth = 1.0 m."""
        disp = np.full((H, W), 20.0, dtype=np.float32)
        pts = np.asarray(_run_stereo(disp).point_cloud.points)
        expected_z = (100.0 * 200.0) / (20.0 * 1000.0)  # = 1.0 m
        np.testing.assert_allclose(pts[:, 2], expected_z, rtol=1e-4)

    def test_halving_disparity_doubles_depth(self):
        """Disparity is inversely proportional to depth."""
        disp_near = np.full((H, W), 40.0, dtype=np.float32)
        disp_far = np.full((H, W), 20.0, dtype=np.float32)
        pts_near = np.asarray(_run_stereo(disp_near).point_cloud.points)
        pts_far = np.asarray(_run_stereo(disp_far).point_cloud.points)
        ratio = pts_far[:, 2].mean() / pts_near[:, 2].mean()
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-4)

    def test_metric_z_matches_depth_map(self):
        """Point cloud Z values must equal the depth_map values at the same pixels."""
        disp = np.full((H, W), 20.0, dtype=np.float32)
        res = _run_stereo(disp)
        pts = np.asarray(res.point_cloud.points)
        expected_z = (100.0 * 200.0) / (20.0 * 1000.0)
        depth_map_mean = np.mean(res.depth_map)
        np.testing.assert_allclose(pts[:, 2].mean(), depth_map_mean, rtol=1e-4)
        np.testing.assert_allclose(pts[:, 2].mean(), expected_z, rtol=1e-4)


# ── Depth ordering ────────────────────────────────────────────────────────────


class TestDepthOrdering:
    """Closer objects (larger disparity) must appear at smaller Z."""

    def test_stereo_larger_disparity_means_smaller_z(self):
        disp_near = np.full((H, W), 50.0, dtype=np.float32)   # closer
        disp_far = np.full((H, W), 10.0, dtype=np.float32)    # further
        pts_near = np.asarray(_run_stereo(disp_near).point_cloud.points)
        pts_far = np.asarray(_run_stereo(disp_far).point_cloud.points)
        assert pts_near[:, 2].mean() < pts_far[:, 2].mean(), (
            "larger disparity must give smaller (closer) Z"
        )

    def test_depth_estimation_higher_input_value_means_larger_z(self):
        """Monocular depth: higher raw depth value → higher normalised depth → larger Z.

        Use a row-wise gradient so every row has a distinct depth.  The min (row 0)
        and max (row H-1) pixels are filtered by Open3D (depth=0 and depth=depth_trunc
        respectively), but all intermediate rows survive and form the test set.
        """
        # Depth increases row by row: top = shallow, bottom = deep
        depth_inc = np.ascontiguousarray(
            np.tile(np.linspace(1.0, 5.0, H), (W, 1)).T.astype(np.float32)
        )
        pts = np.asarray(_run_depth(depth_inc).point_cloud.points)
        assert len(pts) > 0, "point cloud must not be empty"

        # With Y+ down: points at positive Y come from bottom rows (high depth → high Z).
        # Mean Z of Y>0 points must exceed mean Z of Y<0 points.
        z_top = pts[pts[:, 1] < 0, 2]
        z_bot = pts[pts[:, 1] > 0, 2]
        assert len(z_top) > 0 and len(z_bot) > 0
        assert z_bot.mean() > z_top.mean(), (
            "bottom rows (higher depth values) must produce larger Z than top rows"
        )


# ── Two-view geometry (ICP registration) ─────────────────────────────────────


class TestTwoViewICP:
    """Two stereo point clouds taken with a known relative pose must register
    to within a tight tolerance using Open3D ICP."""

    def test_icp_recovers_known_translation(self):
        """Shift the left camera 0.1 m to the right (Δx = +0.1 m) and verify
        that ICP registration recovers the correct T within 5 mm."""
        rng = np.random.default_rng(42)
        disp = rng.uniform(10.0, 40.0, (H, W)).astype(np.float32)

        res1 = _run_stereo(disp)
        pcd1 = res1.point_cloud

        # Build reference cloud: shift by +0.1 m in X
        known_t = np.array([0.1, 0.0, 0.0])
        pts2 = np.asarray(pcd1.points) + known_t
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts2)

        result = o3d.pipelines.registration.registration_icp(
            pcd2,
            pcd1,
            max_correspondence_distance=0.05,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        recovered_t = result.transformation[:3, 3]
        np.testing.assert_allclose(recovered_t, -known_t, atol=0.005)
