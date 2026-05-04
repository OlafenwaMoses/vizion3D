"""
Unit tests for StereoDepthAdvancedConfig and its propagation through
the command, handler, REST API, and gRPC server.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vizion3d.stereo.commands import StereoDepthCommand
from vizion3d.stereo.models import StereoDepthAdvancedConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_stereo_bytes():
    """Return a pair of identical small PNG images as bytes."""
    img = Image.new("RGB", (64, 48), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    return data, data  # (left_bytes, right_bytes)


@pytest.fixture
def fake_disp():
    """Small synthetic disparity map."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.5, 50.0, (48, 64)).astype(np.float32)


# ── StereoDepthAdvancedConfig ─────────────────────────────────────────────────


class TestStereoDepthAdvancedConfig:
    def test_defaults(self):
        cfg = StereoDepthAdvancedConfig()
        assert cfg.focal_length == 1000.0
        assert cfg.cx == 640.0
        assert cfg.cy == 360.0
        assert cfg.baseline == 100.0
        assert cfg.doffs == 0.0
        assert cfg.z_far == 10.0
        assert cfg.conf_threshold == 0.1
        assert cfg.occ_threshold == 0.5
        assert cfg.scale_factor == 1.0

    def test_partial_override_keeps_other_defaults(self):
        cfg = StereoDepthAdvancedConfig(focal_length=1733.74, baseline=536.62)
        assert cfg.focal_length == pytest.approx(1733.74)
        assert cfg.baseline == pytest.approx(536.62)
        assert cfg.cx == 640.0  # unchanged
        assert cfg.cy == 360.0  # unchanged

    def test_all_fields_overridable(self):
        cfg = StereoDepthAdvancedConfig(
            focal_length=800.0,
            cx=320.0,
            cy=240.0,
            baseline=200.0,
            doffs=5.0,
            z_far=15.0,
            conf_threshold=0.3,
            occ_threshold=0.7,
            scale_factor=0.5,
        )
        assert cfg.focal_length == 800.0
        assert cfg.cx == 320.0
        assert cfg.cy == 240.0
        assert cfg.baseline == 200.0
        assert cfg.doffs == 5.0
        assert cfg.z_far == 15.0
        assert cfg.conf_threshold == 0.3
        assert cfg.occ_threshold == 0.7
        assert cfg.scale_factor == 0.5

    def test_invalid_type_rejected(self):
        with pytest.raises(Exception):
            StereoDepthAdvancedConfig(focal_length="not_a_float")


# ── StereoDepthCommand advanced_config field ──────────────────────────────────


class TestStereoDepthCommandAdvancedConfig:
    def test_default_advanced_config_is_correct_type(self, dummy_stereo_bytes):
        left, right = dummy_stereo_bytes
        cmd = StereoDepthCommand(left_image=left, right_image=right)
        assert isinstance(cmd.advanced_config, StereoDepthAdvancedConfig)

    def test_default_advanced_config_has_expected_values(self, dummy_stereo_bytes):
        left, right = dummy_stereo_bytes
        cmd = StereoDepthCommand(left_image=left, right_image=right)
        assert cmd.advanced_config.focal_length == 1000.0
        assert cmd.advanced_config.baseline == 100.0

    def test_each_command_gets_separate_config_instance(self, dummy_stereo_bytes):
        left, right = dummy_stereo_bytes
        cmd1 = StereoDepthCommand(left_image=left, right_image=right)
        cmd2 = StereoDepthCommand(left_image=left, right_image=right)
        assert cmd1.advanced_config is not cmd2.advanced_config

    def test_mutation_of_one_command_does_not_affect_another(self, dummy_stereo_bytes):
        left, right = dummy_stereo_bytes
        cmd1 = StereoDepthCommand(left_image=left, right_image=right)
        cmd2 = StereoDepthCommand(left_image=left, right_image=right)
        cmd1.advanced_config.focal_length = 999.0
        assert cmd2.advanced_config.focal_length == 1000.0

    def test_custom_config_stored_on_command(self, dummy_stereo_bytes):
        left, right = dummy_stereo_bytes
        cfg = StereoDepthAdvancedConfig(focal_length=1733.74, baseline=536.62)
        cmd = StereoDepthCommand(left_image=left, right_image=right, advanced_config=cfg)
        assert cmd.advanced_config.focal_length == pytest.approx(1733.74)
        assert cmd.advanced_config.baseline == pytest.approx(536.62)


# ── Depth formula correctness ─────────────────────────────────────────────────


class TestDepthFormula:
    """Validate the disparity→depth formula: depth_m = baseline * focal / disp / 1000."""

    def test_depth_from_known_disparity(self):
        # baseline=100mm, focal=1000px, disp=10px → depth = 100*1000/10/1000 = 10m
        baseline = 100.0
        focal = 1000.0
        disp = 10.0
        expected_depth = baseline * focal / disp / 1000.0
        assert expected_depth == pytest.approx(10.0)

    def test_zero_disparity_gives_zero_depth(self):
        # Handler sets depth_mm[disp<=0]=0.0, so depth_m=0
        disp = np.array([0.0, 1.0, 2.0])
        baseline = 100.0
        focal = 1000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_mm = baseline * focal / (disp + 0.0)
        depth_mm[disp <= 0] = 0.0
        depth_m = depth_mm / 1000.0
        assert depth_m[0] == 0.0

    def test_doffs_shifts_depth(self):
        # doffs shifts the effective disparity: depth = baseline * focal / (disp + doffs)
        baseline = 100.0
        focal = 1000.0
        disp = 10.0
        doffs = 5.0
        depth = baseline * focal / (disp + doffs) / 1000.0
        expected = 100.0 * 1000.0 / 15.0 / 1000.0
        assert depth == pytest.approx(expected)


# ── REST API propagates config form fields ────────────────────────────────────


class TestStereoRestAdvancedConfig:
    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("open3d", reason="open3d required — run: uv python pin 3.12 && uv sync")
        from fastapi.testclient import TestClient

        from vizion3d.server.rest.app import app

        self.client = TestClient(app)

    def _post(self, extra_data=None):
        img = Image.new("RGB", (32, 32), color="green")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        result = MagicMock()
        result.depth_map = [[1.0]]
        result.disparity_map = [[5.0]]
        result.min_depth = 1.0
        result.max_depth = 1.0
        result.backend_used = "/fake/stereo.pth"
        result.depth_image = None
        result.point_cloud = None
        result.mesh = None

        with patch("vizion3d.server.rest.stereo_depth.StereoDepth") as mock_cls:
            mock_cls.return_value.run.return_value = result
            self.client.post(
                "/lifting/stereo-depth",
                files={
                    "left_image": ("left.png", io.BytesIO(img_bytes), "image/png"),
                    "right_image": ("right.png", io.BytesIO(img_bytes), "image/png"),
                },
                data=extra_data or {},
            )
            return mock_cls.return_value.run.call_args[0][0]

    def test_default_config_used_when_no_form_fields_sent(self):
        cmd = self._post()
        assert cmd.advanced_config.focal_length == 1000.0
        assert cmd.advanced_config.baseline == 100.0
        assert cmd.advanced_config.cx == 640.0
        assert cmd.advanced_config.cy == 360.0
        assert cmd.advanced_config.z_far == 10.0

    def test_custom_focal_length_forwarded(self):
        cmd = self._post({"focal_length": "1733.74"})
        assert cmd.advanced_config.focal_length == pytest.approx(1733.74)
        assert cmd.advanced_config.baseline == 100.0  # unchanged

    def test_custom_baseline_forwarded(self):
        cmd = self._post({"baseline": "536.62"})
        assert cmd.advanced_config.baseline == pytest.approx(536.62)

    def test_custom_cx_cy_forwarded(self):
        cmd = self._post({"cx": "792.27", "cy": "541.89"})
        assert cmd.advanced_config.cx == pytest.approx(792.27)
        assert cmd.advanced_config.cy == pytest.approx(541.89)

    def test_custom_z_far_forwarded(self):
        cmd = self._post({"z_far": "5.0"})
        assert cmd.advanced_config.z_far == pytest.approx(5.0)

    def test_custom_scale_factor_forwarded(self):
        cmd = self._post({"scale_factor": "0.5"})
        assert cmd.advanced_config.scale_factor == pytest.approx(0.5)

    def test_partial_override_does_not_affect_other_fields(self):
        cmd = self._post({"focal_length": "800.0"})
        assert cmd.advanced_config.focal_length == pytest.approx(800.0)
        assert cmd.advanced_config.cy == 360.0  # unchanged


# ── gRPC server propagates config proto fields ────────────────────────────────


class TestStereoGrpcAdvancedConfig:
    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("open3d", reason="open3d required — run: uv python pin 3.12 && uv sync")
        from vizion3d.proto import lifting_pb2
        from vizion3d.server.grpc.server import LiftingServiceServicer

        self.pb2 = lifting_pb2
        self.servicer = LiftingServiceServicer()
        self.context = MagicMock()

        img = Image.new("RGB", (32, 32), color=(50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.image_bytes = buf.getvalue()

    def _run(self, proto_cfg=None):
        kwargs = {
            "left_image_bytes": self.image_bytes,
            "right_image_bytes": self.image_bytes,
        }
        if proto_cfg is not None:
            kwargs["advanced_config"] = proto_cfg
        request = self.pb2.StereoDepthRequest(**kwargs)

        result = MagicMock()
        result.depth_map = [[1.0]]
        result.disparity_map = [[5.0]]
        result.min_depth = 1.0
        result.max_depth = 1.0
        result.backend_used = "/fake/stereo.pth"
        result.depth_image = None
        result.point_cloud = None
        result.mesh = None

        with patch("vizion3d.server.grpc.server.StereoDepth") as mock_cls:
            mock_cls.return_value.run.return_value = result
            self.servicer.RunStereoDepth(request, self.context)
            return mock_cls.return_value.run.call_args[0][0]

    def test_no_config_in_request_uses_defaults(self):
        cmd = self._run()
        assert cmd.advanced_config.focal_length == 1000.0
        assert cmd.advanced_config.baseline == 100.0

    def test_full_config_forwarded(self):
        proto_cfg = self.pb2.StereoDepthAdvancedConfig(
            focal_length=1733.74,
            cx=792.27,
            cy=541.89,
            baseline=536.62,
            doffs=0.0,
            z_far=8.0,
            conf_threshold=0.2,
            occ_threshold=0.6,
            scale_factor=0.5,
        )
        cmd = self._run(proto_cfg)
        assert cmd.advanced_config.focal_length == pytest.approx(1733.74)
        assert cmd.advanced_config.cx == pytest.approx(792.27)
        assert cmd.advanced_config.cy == pytest.approx(541.89)
        assert cmd.advanced_config.baseline == pytest.approx(536.62)
        assert cmd.advanced_config.z_far == pytest.approx(8.0)
        assert cmd.advanced_config.conf_threshold == pytest.approx(0.2)
        assert cmd.advanced_config.occ_threshold == pytest.approx(0.6)
        assert cmd.advanced_config.scale_factor == pytest.approx(0.5)

    def test_partial_config_overrides_only_set_fields(self):
        proto_cfg = self.pb2.StereoDepthAdvancedConfig(focal_length=800.0, z_far=5.0)
        cmd = self._run(proto_cfg)
        assert cmd.advanced_config.focal_length == pytest.approx(800.0)
        assert cmd.advanced_config.baseline == 100.0  # default
        assert cmd.advanced_config.cx == 640.0  # default
        assert cmd.advanced_config.z_far == pytest.approx(5.0)

    def test_empty_config_message_uses_all_defaults(self):
        proto_cfg = self.pb2.StereoDepthAdvancedConfig()
        cmd = self._run(proto_cfg)
        assert cmd.advanced_config.focal_length == 1000.0
        assert cmd.advanced_config.baseline == 100.0
