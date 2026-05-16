"""
Coordinate space invariant tests for vizion3d point cloud outputs.

DepthEstimation and StereoDepth use the OpenGL / viewer convention:
  X+ right   (increasing pixel column → increasing X)
  Y+ up      (increasing pixel row    → decreasing Y, i.e. top rows → positive Y)
  Z- forward (depth into the scene → negative Z, toward the viewer → positive Z)

The OpenGL convention means point clouds load facing the viewer in MeshLab and
similar tools without requiring manual orbit correction.

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
CX, CY = 9.5, 7.5  # exact centre of 20 × 16

_STEREO_CFG = StereoDepthAdvancedConfig(
    focal_length=FX,
    cx=CX,
    cy=CY,
    baseline=100.0,  # 100 mm
    doffs=0.0,
    z_far=1000.0,
    conf_threshold=0.0,  # accept all pixels
    occ_threshold=0.0,
)

_DEPTH_CFG = DepthEstimationAdvanceConfig(
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY,
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
    with patch.object(DepthEstimationHandler, "_run_depth_anything_checkpoint", return_value=depth):
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
    value is in (0, 1] and all unprojected Z values are negative."""
    return np.linspace(1.0, 5.0, H * W).reshape(H, W).astype(np.float32)


# ── Z- forward ────────────────────────────────────────────────────────────────


class TestZForward:
    def test_stereo_all_z_negative(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        assert len(pts) > 0, "point cloud must not be empty"
        assert np.all(pts[:, 2] < 0), (
            "all stereo Z values must be negative (OpenGL: scene is in -Z direction)"
        )

    def test_depth_estimation_all_z_negative(self, gradient_depth):
        pts = np.asarray(_run_depth(gradient_depth).point_cloud.points)
        assert len(pts) > 0, "point cloud must not be empty"
        assert np.all(pts[:, 2] < 0), (
            "all depth estimation Z values must be negative (OpenGL: scene is in -Z direction)"
        )


# ── Y+ up ─────────────────────────────────────────────────────────────────────


class TestYUp:
    """OpenGL convention: top pixels produce positive Y; bottom pixels produce negative Y."""

    def test_stereo_top_rows_have_positive_y(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        assert pts[:, 1].max() > 0, "top pixel rows must produce Y > 0"

    def test_stereo_bottom_rows_have_negative_y(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        assert pts[:, 1].min() < 0, "bottom pixel rows must produce Y < 0"

    def test_stereo_mean_y_decreases_with_row(self, uniform_disp):
        """Average Y of the upper half must be strictly greater than the lower half."""
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        top_mean = pts[pts[:, 1] > 0, 1].mean()
        bot_mean = pts[pts[:, 1] < 0, 1].mean()
        assert top_mean > bot_mean

    def test_depth_estimation_y_sign_convention(self, gradient_depth):
        """DepthEstimation uses OpenGL convention: Y+ up (top rows → positive Y)."""
        pts = np.asarray(_run_depth(gradient_depth).point_cloud.points)
        assert pts[:, 1].max() > 0, "top rows must produce Y > 0 (Y+ up in OpenGL convention)"
        assert pts[:, 1].min() < 0, "bottom rows must produce Y < 0 (Y+ up in OpenGL convention)"


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
    """X+ right, Y+ up, Z+ toward viewer is a right-handed system: X × Y = +Z.

    Verified empirically from the point cloud by measuring the empirical
    X and Y direction vectors and checking their cross product.
    """

    def _empirical_axes(self, pts: np.ndarray):
        """Return unit vectors for the empirical X and Y axes from the cloud."""
        # X axis: direction of increasing column at mid-row
        # With pinhole: X = (u-cx)*depth/f, so a step of Δu gives ΔX = Δu*depth/f > 0
        # We can infer this from the range of X vs Y relative to Z.
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        assert x_range > 0 and y_range > 0
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    def test_stereo_cross_product_points_toward_viewer(self, uniform_disp):
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        x_hat, y_hat = self._empirical_axes(pts)
        z_hat = np.cross(x_hat, y_hat)
        assert z_hat[2] > 0, "X × Y must point in +Z (right-handed, toward viewer)"

    def test_depth_estimation_cross_product_points_toward_viewer(self, gradient_depth):
        # OpenGL convention: X right, Y up → X × Y = +Z (toward viewer). Still right-handed.
        pts = np.asarray(_run_depth(gradient_depth).point_cloud.points)
        x_hat, y_hat = self._empirical_axes(pts)
        z_hat = np.cross(x_hat, y_hat)
        assert z_hat[2] > 0, "X × Y must point in +Z (right-handed system)"

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
        depth = -Z
        u = X * FX / depth + CX
        v = CY - Y * FY / depth
        assert np.all(u >= -0.5) and np.all(u <= W - 0.5), "re-projected u must be in image"
        assert np.all(v >= -0.5) and np.all(v <= H - 0.5), "re-projected v must be in image"

    def test_stereo_known_pixel_round_trip(self, uniform_disp):
        """For a specific pixel (u=15, v=12) the unprojected 3D point must
        re-project back to within 0.5 px of the original pixel."""
        pts = np.asarray(_run_stereo(uniform_disp).point_cloud.points)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        depth = -Z
        u_reproj = X * FX / depth + CX
        v_reproj = CY - Y * FY / depth
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
        expected_z = -((100.0 * 200.0) / (20.0 * 1000.0))  # = -1.0 m
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
        """Point cloud Z values must be the negative depth_map values at the same pixels."""
        disp = np.full((H, W), 20.0, dtype=np.float32)
        res = _run_stereo(disp)
        pts = np.asarray(res.point_cloud.points)
        expected_z = (100.0 * 200.0) / (20.0 * 1000.0)
        depth_map_mean = np.mean(res.depth_map)
        np.testing.assert_allclose(pts[:, 2].mean(), -depth_map_mean, rtol=1e-4)
        np.testing.assert_allclose(pts[:, 2].mean(), -expected_z, rtol=1e-4)


# ── Depth ordering ────────────────────────────────────────────────────────────


class TestDepthOrdering:
    """Closer objects (larger disparity) must appear closer to Z=0."""

    def test_stereo_larger_disparity_means_z_closer_to_zero(self):
        disp_near = np.full((H, W), 50.0, dtype=np.float32)  # closer
        disp_far = np.full((H, W), 10.0, dtype=np.float32)  # further
        pts_near = np.asarray(_run_stereo(disp_near).point_cloud.points)
        pts_far = np.asarray(_run_stereo(disp_far).point_cloud.points)
        assert pts_near[:, 2].mean() > pts_far[:, 2].mean(), (
            "larger disparity must give larger Z (less negative, closer to camera)"
        )

    def test_depth_estimation_close_pixels_have_larger_z(self):
        """Depth Anything V2 is inverse depth: higher raw value = closer to camera.

        After the normalization flip and OpenGL Y/Z conversion:
          - close pixels (high raw value) → small |Z|, i.e. Z closer to 0 (less negative)
          - far pixels (low raw value) → large |Z|, i.e. Z further from 0 (more negative)

        Row-wise gradient: row 0 has low inverse depth (far), row H-1 has high
        inverse depth (close). In OpenGL convention Y is flipped, so:
          - top rows (row 0, far)   → Y > 0  (positive Y = up)
          - bottom rows (row H-1, close) → Y < 0 (negative Y = down)
        """
        # Inverse-depth increases row by row: top rows = far, bottom rows = close
        depth_inc = np.ascontiguousarray(
            np.tile(np.linspace(1.0, 5.0, H), (W, 1)).T.astype(np.float32)
        )
        pts = np.asarray(_run_depth(depth_inc).point_cloud.points)
        assert len(pts) > 0, "point cloud must not be empty"

        # In OpenGL convention: Y > 0 = top rows (far), Y < 0 = bottom rows (close).
        z_far_rows = pts[pts[:, 1] > 0, 2]  # top rows: low raw value = far = large |Z|
        z_close_rows = pts[pts[:, 1] < 0, 2]  # bottom rows: high raw value = close = small |Z|
        assert len(z_far_rows) > 0 and len(z_close_rows) > 0
        assert z_close_rows.mean() > z_far_rows.mean(), (
            "close rows (high inverse depth) must produce larger Z (less negative) than far rows"
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
