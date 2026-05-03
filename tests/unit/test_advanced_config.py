"""
Unit tests for DepthEstimationAdvanceConfig and its propagation through
the command, handler, REST API, and gRPC server.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vizion3d.lifting.commands import DepthEstimationCommand
from vizion3d.lifting.handlers import DepthEstimationHandler
from vizion3d.lifting.models import DepthEstimationAdvanceConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_depth():
    rng = np.random.default_rng(0)
    return rng.uniform(0.5, 3.0, (100, 100))


# ── DepthEstimationAdvanceConfig ──────────────────────────────────────────────

class TestDepthEstimationAdvanceConfig:
    def test_defaults_match_primesense(self):
        cfg = DepthEstimationAdvanceConfig()
        assert cfg.fx == 525.0
        assert cfg.fy == 525.0
        assert cfg.cx == 319.5
        assert cfg.cy == 239.5
        assert cfg.depth_scale == 1000.0
        assert cfg.depth_trunc == 10.0

    def test_partial_override_keeps_other_defaults(self):
        cfg = DepthEstimationAdvanceConfig(fx=615.0, fy=615.0)
        assert cfg.fx == 615.0
        assert cfg.fy == 615.0
        assert cfg.cx == 319.5
        assert cfg.cy == 239.5
        assert cfg.depth_scale == 1000.0
        assert cfg.depth_trunc == 10.0

    def test_all_fields_overridable(self):
        cfg = DepthEstimationAdvanceConfig(
            fx=700.0, fy=701.0, cx=320.0, cy=241.0,
            depth_scale=500.0, depth_trunc=5.0,
        )
        assert cfg.fx == 700.0
        assert cfg.fy == 701.0
        assert cfg.cx == 320.0
        assert cfg.cy == 241.0
        assert cfg.depth_scale == 500.0
        assert cfg.depth_trunc == 5.0

    def test_invalid_type_rejected(self):
        with pytest.raises(Exception):
            DepthEstimationAdvanceConfig(fx="not_a_number")


# ── DepthEstimationCommand advanced_config field ──────────────────────────────

class TestDepthEstimationCommandAdvancedConfig:
    def test_default_advanced_config_is_correct_type(self):
        cmd = DepthEstimationCommand(image_input=b"img")
        assert isinstance(cmd.advanced_config, DepthEstimationAdvanceConfig)

    def test_default_advanced_config_has_primesense_values(self):
        cmd = DepthEstimationCommand(image_input=b"img")
        assert cmd.advanced_config.fx == 525.0
        assert cmd.advanced_config.depth_trunc == 10.0

    def test_each_command_gets_separate_config_instance(self):
        cmd1 = DepthEstimationCommand(image_input=b"img")
        cmd2 = DepthEstimationCommand(image_input=b"img")
        assert cmd1.advanced_config is not cmd2.advanced_config

    def test_mutation_of_one_command_does_not_affect_another(self):
        cmd1 = DepthEstimationCommand(image_input=b"img")
        cmd2 = DepthEstimationCommand(image_input=b"img")
        cmd1.advanced_config.fx = 999.0
        assert cmd2.advanced_config.fx == 525.0

    def test_custom_config_stored_on_command(self):
        cfg = DepthEstimationAdvanceConfig(fx=615.0, depth_trunc=5.0)
        cmd = DepthEstimationCommand(image_input=b"img", advanced_config=cfg)
        assert cmd.advanced_config.fx == 615.0
        assert cmd.advanced_config.depth_trunc == 5.0


# ── _depth_array_to_rgbd_depth ────────────────────────────────────────────────

class TestDepthArrayToRgbdDepth:
    def test_output_dtype_is_uint16(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
        result = DepthEstimationHandler._depth_array_to_rgbd_depth(arr, 1000.0, 10.0)
        assert result.dtype == np.uint16

    def test_zero_range_input_returns_zeros(self):
        arr = np.ones((4, 4), dtype=float) * 3.0
        result = DepthEstimationHandler._depth_array_to_rgbd_depth(arr, 1000.0, 10.0)
        assert np.all(result == 0)

    def test_max_value_equals_trunc_times_scale(self):
        arr = np.array([[0.0, 1.0]], dtype=float)
        result = DepthEstimationHandler._depth_array_to_rgbd_depth(arr, 1000.0, 10.0)
        assert result[0, 1] == pytest.approx(10_000, abs=1)

    def test_custom_depth_scale_halves_output(self):
        arr = np.array([[0.0, 1.0]], dtype=float)
        full = DepthEstimationHandler._depth_array_to_rgbd_depth(arr, 1000.0, 10.0)
        half = DepthEstimationHandler._depth_array_to_rgbd_depth(arr, 500.0, 10.0)
        assert half[0, 1] == pytest.approx(full[0, 1] / 2, abs=1)

    def test_custom_depth_trunc_halves_output(self):
        arr = np.array([[0.0, 1.0]], dtype=float)
        full = DepthEstimationHandler._depth_array_to_rgbd_depth(arr, 1000.0, 10.0)
        half = DepthEstimationHandler._depth_array_to_rgbd_depth(arr, 1000.0, 5.0)
        assert half[0, 1] == pytest.approx(full[0, 1] / 2, abs=1)

    def test_output_clipped_to_uint16_max(self):
        arr = np.array([[0.0, 1.0]], dtype=float)
        result = DepthEstimationHandler._depth_array_to_rgbd_depth(arr, 1e6, 1e6)
        assert result[0, 1] == np.iinfo(np.uint16).max

    def test_min_value_is_always_zero(self):
        arr = np.array([[2.0, 5.0, 10.0]], dtype=float)
        result = DepthEstimationHandler._depth_array_to_rgbd_depth(arr, 1000.0, 10.0)
        assert result[0, 0] == 0


# ── Handler propagates config to Open3D ──────────────────────────────────────

class TestHandlerPropagatesAdvancedConfig:
    def test_custom_intrinsics_forwarded_to_pinhole(self, dummy_image_bytes, fake_depth):
        open3d = pytest.importorskip(
            "open3d", reason="open3d required — run: uv python pin 3.12 && uv sync"
        )
        cfg = DepthEstimationAdvanceConfig(fx=615.0, fy=616.0, cx=321.0, cy=241.0)
        captured = []
        original_cls = open3d.camera.PinholeCameraIntrinsic

        def capturing(*args, **kwargs):
            captured.append(args)
            return original_cls(*args, **kwargs)

        with (
            patch.object(
                DepthEstimationHandler,
                "_run_depth_anything_checkpoint",
                return_value=fake_depth,
            ),
            patch.object(open3d.camera, "PinholeCameraIntrinsic", side_effect=capturing),
        ):
            DepthEstimationHandler().handle(
                DepthEstimationCommand(
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                    return_point_cloud=True,
                    advanced_config=cfg,
                )
            )

        assert len(captured) == 1
        _w, _h, fx, fy, cx, cy = captured[0]
        assert fx == 615.0
        assert fy == 616.0
        assert cx == 321.0
        assert cy == 241.0

    def test_custom_depth_scale_trunc_forwarded_to_rgbd(self, dummy_image_bytes, fake_depth):
        open3d = pytest.importorskip(
            "open3d", reason="open3d required — run: uv python pin 3.12 && uv sync"
        )
        cfg = DepthEstimationAdvanceConfig(depth_scale=500.0, depth_trunc=3.0)
        captured_kwargs = {}
        original_fn = open3d.geometry.RGBDImage.create_from_color_and_depth

        def capturing(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_fn(*args, **kwargs)

        with (
            patch.object(
                DepthEstimationHandler,
                "_run_depth_anything_checkpoint",
                return_value=fake_depth,
            ),
            patch.object(
                open3d.geometry.RGBDImage,
                "create_from_color_and_depth",
                side_effect=capturing,
            ),
        ):
            DepthEstimationHandler().handle(
                DepthEstimationCommand(
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pth",
                    return_point_cloud=True,
                    advanced_config=cfg,
                )
            )

        assert captured_kwargs["depth_scale"] == 500.0
        assert captured_kwargs["depth_trunc"] == 3.0


# ── REST API propagates config form fields ────────────────────────────────────

class TestRestAdvancedConfig:
    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip(
            "open3d", reason="open3d required — run: uv python pin 3.12 && uv sync"
        )
        from fastapi.testclient import TestClient

        from vizion3d.server.rest.app import app
        self.client = TestClient(app)

    def _post(self, extra_data=None):
        img = Image.new("RGB", (50, 50), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        result = MagicMock()
        result.depth_map = [[1.0]]
        result.min_depth = 1.0
        result.max_depth = 1.0
        result.backend_used = "/fake/model.pth"
        result.depth_image = None
        result.point_cloud = None
        result.mesh = None

        with patch("vizion3d.server.rest.app.DepthEstimation") as mock_cls:
            mock_cls.return_value.run.return_value = result
            self.client.post(
                "/lifting/depth-estimation",
                files={"image": ("test.png", buf, "image/png")},
                data=extra_data or {},
            )
            return mock_cls.return_value.run.call_args[0][0]

    def test_default_config_used_when_no_form_fields_sent(self):
        cmd = self._post()
        assert cmd.advanced_config.fx == 525.0
        assert cmd.advanced_config.fy == 525.0
        assert cmd.advanced_config.cx == 319.5
        assert cmd.advanced_config.cy == 239.5
        assert cmd.advanced_config.depth_scale == 1000.0
        assert cmd.advanced_config.depth_trunc == 10.0

    def test_custom_fx_fy_forwarded(self):
        cmd = self._post({"fx": "615.0", "fy": "616.0"})
        assert cmd.advanced_config.fx == pytest.approx(615.0)
        assert cmd.advanced_config.fy == pytest.approx(616.0)
        assert cmd.advanced_config.cx == 319.5  # unchanged

    def test_custom_cx_cy_forwarded(self):
        cmd = self._post({"cx": "321.0", "cy": "241.0"})
        assert cmd.advanced_config.cx == pytest.approx(321.0)
        assert cmd.advanced_config.cy == pytest.approx(241.0)

    def test_custom_depth_scale_forwarded(self):
        cmd = self._post({"depth_scale": "500.0"})
        assert cmd.advanced_config.depth_scale == pytest.approx(500.0)
        assert cmd.advanced_config.depth_trunc == 10.0  # unchanged

    def test_custom_depth_trunc_forwarded(self):
        cmd = self._post({"depth_trunc": "5.0"})
        assert cmd.advanced_config.depth_trunc == pytest.approx(5.0)
        assert cmd.advanced_config.depth_scale == 1000.0  # unchanged

    def test_partial_override_does_not_affect_other_fields(self):
        cmd = self._post({"fx": "700.0"})
        assert cmd.advanced_config.fx == pytest.approx(700.0)
        assert cmd.advanced_config.fy == 525.0
        assert cmd.advanced_config.depth_trunc == 10.0


# ── gRPC server propagates config proto fields ────────────────────────────────

class TestGrpcAdvancedConfig:
    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip(
            "open3d", reason="open3d required — run: uv python pin 3.12 && uv sync"
        )
        from vizion3d.proto import lifting_pb2
        from vizion3d.server.grpc.server import LiftingServiceServicer
        self.pb2 = lifting_pb2
        self.servicer = LiftingServiceServicer()
        self.context = MagicMock()

        img = Image.new("RGB", (50, 50), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.image_bytes = buf.getvalue()

    def _run(self, proto_cfg=None):
        kwargs = {"image_bytes": self.image_bytes}
        if proto_cfg is not None:
            kwargs["advanced_config"] = proto_cfg
        request = self.pb2.DepthEstimationRequest(**kwargs)

        result = MagicMock()
        result.depth_map = [[1.0]]
        result.min_depth = 1.0
        result.max_depth = 1.0
        result.backend_used = "/fake/model.pth"
        result.depth_image = None
        result.point_cloud = None
        result.mesh = None

        with patch("vizion3d.server.grpc.server.DepthEstimation") as mock_cls:
            mock_cls.return_value.run.return_value = result
            self.servicer.RunDepthEstimation(request, self.context)
            return mock_cls.return_value.run.call_args[0][0]

    def test_no_config_in_request_uses_defaults(self):
        cmd = self._run()
        assert cmd.advanced_config.fx == 525.0
        assert cmd.advanced_config.depth_trunc == 10.0

    def test_full_config_forwarded(self):
        proto_cfg = self.pb2.DepthEstimationAdvanceConfig(
            fx=615.0, fy=616.0, cx=320.0, cy=241.0,
            depth_scale=500.0, depth_trunc=3.0,
        )
        cmd = self._run(proto_cfg)
        assert cmd.advanced_config.fx == pytest.approx(615.0)
        assert cmd.advanced_config.fy == pytest.approx(616.0)
        assert cmd.advanced_config.cx == pytest.approx(320.0)
        assert cmd.advanced_config.cy == pytest.approx(241.0)
        assert cmd.advanced_config.depth_scale == pytest.approx(500.0)
        assert cmd.advanced_config.depth_trunc == pytest.approx(3.0)

    def test_partial_config_overrides_only_set_fields(self):
        proto_cfg = self.pb2.DepthEstimationAdvanceConfig(fx=700.0, depth_trunc=2.0)
        cmd = self._run(proto_cfg)
        assert cmd.advanced_config.fx == pytest.approx(700.0)
        assert cmd.advanced_config.fy == 525.0       # default
        assert cmd.advanced_config.cx == 319.5       # default
        assert cmd.advanced_config.depth_scale == 1000.0  # default
        assert cmd.advanced_config.depth_trunc == pytest.approx(2.0)

    def test_empty_config_message_uses_all_defaults(self):
        proto_cfg = self.pb2.DepthEstimationAdvanceConfig()
        cmd = self._run(proto_cfg)
        assert cmd.advanced_config.fx == 525.0
        assert cmd.advanced_config.fy == 525.0
        assert cmd.advanced_config.depth_trunc == 10.0
