"""
Unit tests for StereoDepthHandler internals and the StereoDepth facade.

These tests mock the S2M2 model so no checkpoint file is needed on disk.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vizion3d.stereo.commands import StereoDepthCommand
from vizion3d.stereo.handlers import StereoDepthHandler
from vizion3d.stereo.models import StereoDepthAdvancedConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_image_bytes():
    img = Image.new("RGB", (64, 48), color=(80, 120, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_disp():
    """Realistic disparity field: positive, 48×64 float32 array."""
    rng = np.random.default_rng(7)
    return rng.uniform(1.0, 50.0, (48, 64)).astype(np.float32)


# ── Handler device helper ─────────────────────────────────────────────────────


class TestHandlerDeviceHelper:
    def test_returns_cuda_when_available(self):
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        assert StereoDepthHandler._torch_device(torch_mock) == "cuda"

    def test_returns_mps_when_cuda_unavailable(self):
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        torch_mock.backends.mps.is_available.return_value = True
        assert StereoDepthHandler._torch_device(torch_mock) == "mps"

    def test_returns_cpu_as_fallback(self):
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        torch_mock.backends.mps.is_available.return_value = False
        assert StereoDepthHandler._torch_device(torch_mock) == "cpu"


# ── Handler output shapes ─────────────────────────────────────────────────────


class TestHandlerOutputShapes:
    """Verify depth_map, disparity_map shapes and metric depth computation."""

    def test_depth_map_is_2d_list_of_floats(self, dummy_image_bytes, fake_disp):
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=fake_disp):
            cmd = StereoDepthCommand(
                left_image=dummy_image_bytes,
                right_image=dummy_image_bytes,
                model_backend="/fake/model.pth",
            )
            result = StereoDepthHandler().handle(cmd)

        assert isinstance(result.depth_map, list)
        assert isinstance(result.depth_map[0], list)
        assert isinstance(result.depth_map[0][0], float)

    def test_disparity_map_matches_input_shape(self, dummy_image_bytes, fake_disp):
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=fake_disp):
            cmd = StereoDepthCommand(
                left_image=dummy_image_bytes,
                right_image=dummy_image_bytes,
                model_backend="/fake/model.pth",
            )
            result = StereoDepthHandler().handle(cmd)

        H, W = fake_disp.shape
        assert len(result.disparity_map) == H
        assert len(result.disparity_map[0]) == W

    def test_min_max_depth_ordering(self, dummy_image_bytes, fake_disp):
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=fake_disp):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                )
            )
        assert result.max_depth >= result.min_depth

    def test_point_cloud_scale_is_1(self, dummy_image_bytes, fake_disp):
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=fake_disp):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                )
            )
        assert result.point_cloud_scale == 1.0

    def test_backend_used_propagated(self, dummy_image_bytes, fake_disp):
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=fake_disp):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                )
            )
        assert result.backend_used == "/fake/model.pth"


# ── Depth formula integration ─────────────────────────────────────────────────


class TestDepthMetricConversion:
    """Verify the handler converts disparity to real metric depth correctly."""

    def test_high_disparity_gives_low_depth(self, dummy_image_bytes):
        # All pixels at disp=100px: depth = 100 * 1000 / 100 / 1000 = 1m
        uniform_disp = np.full((48, 64), 100.0, dtype=np.float32)
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=uniform_disp):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                    advanced_config=StereoDepthAdvancedConfig(focal_length=1000.0, baseline=100.0),
                )
            )
        # depth = baseline * focal / disp / 1000 = 100*1000/100/1000 = 1.0 m
        assert result.min_depth == pytest.approx(1.0, abs=1e-3)
        assert result.max_depth == pytest.approx(1.0, abs=1e-3)

    def test_zero_disparity_pixels_excluded_from_depth(self, dummy_image_bytes):
        disp = np.zeros((48, 64), dtype=np.float32)
        disp[0, 0] = 10.0  # one valid pixel
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=disp):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                    advanced_config=StereoDepthAdvancedConfig(focal_length=1000.0, baseline=100.0),
                )
            )
        # Zero disparity → zero depth; non-zero depth present for (0,0)
        assert result.min_depth == pytest.approx(0.0)
        assert result.max_depth > 0.0


# ── Optional outputs ──────────────────────────────────────────────────────────


class TestHandlerOptionalOutputs:
    def test_no_optional_outputs_by_default(self, dummy_image_bytes, fake_disp):
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=fake_disp):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                )
            )
        assert result.depth_image is None
        assert result.point_cloud is None
        assert result.mesh is None

    def test_return_depth_image_requires_open3d(self, dummy_image_bytes, fake_disp):
        pytest.importorskip("open3d", reason="open3d required")
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=fake_disp):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                    return_depth_image=True,
                )
            )
        assert result.depth_image is not None
        arr = np.asarray(result.depth_image)
        assert arr.dtype == np.uint16

    def test_return_point_cloud_requires_open3d(self, dummy_image_bytes, fake_disp):
        pytest.importorskip("open3d", reason="open3d required")
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=fake_disp):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                    return_point_cloud=True,
                    advanced_config=StereoDepthAdvancedConfig(
                        focal_length=1000.0, baseline=100.0, z_far=10.0
                    ),
                )
            )
        assert result.point_cloud is not None
        assert result.point_cloud.has_points()
        assert result.point_cloud.has_colors()

    def test_conf_occ_thresholds_filter_point_cloud(self, dummy_image_bytes, fake_disp):
        pytest.importorskip("open3d", reason="open3d required")
        occ = np.ones_like(fake_disp, dtype=np.float32)
        conf = np.ones_like(fake_disp, dtype=np.float32)
        conf[:, :32] = 0.0

        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=(fake_disp, occ, conf)):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                    return_point_cloud=True,
                    advanced_config=StereoDepthAdvancedConfig(
                        focal_length=1000.0,
                        baseline=100.0,
                        z_far=100.0,
                        conf_threshold=0.5,
                        occ_threshold=0.5,
                    ),
                )
            )

        assert result.point_cloud is not None
        assert len(np.asarray(result.point_cloud.points)) < fake_disp.size

    def test_return_mesh_requires_open3d(self, dummy_image_bytes, fake_disp):
        open3d = pytest.importorskip("open3d", reason="open3d required")
        with patch.object(StereoDepthHandler, "_run_s2m2", return_value=fake_disp):
            result = StereoDepthHandler().handle(
                StereoDepthCommand(
                    left_image=dummy_image_bytes,
                    right_image=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                    return_mesh=True,
                    advanced_config=StereoDepthAdvancedConfig(
                        focal_length=1000.0, baseline=100.0, z_far=10.0
                    ),
                )
            )
        assert result.mesh is not None
        assert isinstance(result.mesh, open3d.geometry.TriangleMesh)


# ── S2M2 variant detection ────────────────────────────────────────────────────


class TestS2M2VariantDetection:
    def test_l_variant_detected(self):
        from vizion3d.stereo.arch.s2m2 import s2m2_config_from_checkpoint

        cfg = s2m2_config_from_checkpoint("/models/stereo-depth-s2m2-L.pth")
        assert cfg["feature_channels"] == 256
        assert cfg["num_transformer"] == 3

    def test_s_variant_detected(self):
        from vizion3d.stereo.arch.s2m2 import s2m2_config_from_checkpoint

        cfg = s2m2_config_from_checkpoint("/models/stereo-depth-s2m2-S.pth")
        assert cfg["feature_channels"] == 128
        assert cfg["num_transformer"] == 1

    def test_m_variant_detected(self):
        from vizion3d.stereo.arch.s2m2 import s2m2_config_from_checkpoint

        cfg = s2m2_config_from_checkpoint("/models/stereo-depth-s2m2-M.pth")
        assert cfg["feature_channels"] == 192
        assert cfg["num_transformer"] == 2

    def test_xl_variant_detected(self):
        from vizion3d.stereo.arch.s2m2 import s2m2_config_from_checkpoint

        cfg = s2m2_config_from_checkpoint("/models/stereo-depth-s2m2-XL.pth")
        assert cfg["feature_channels"] == 384
        assert cfg["num_transformer"] == 3

    def test_unknown_variant_falls_back_to_l(self):
        from vizion3d.stereo.arch.s2m2 import s2m2_config_from_checkpoint

        cfg = s2m2_config_from_checkpoint("/models/my-custom-stereo-model.pth")
        assert cfg["feature_channels"] == 256  # L default
        assert cfg["num_transformer"] == 3

    def test_xl_not_confused_with_l(self):
        from vizion3d.stereo.arch.s2m2 import s2m2_config_from_checkpoint

        cfg = s2m2_config_from_checkpoint("/models/model-XL.pth")
        assert cfg["feature_channels"] == 384  # XL, not L
