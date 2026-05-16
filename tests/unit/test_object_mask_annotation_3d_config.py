"""
Unit tests for ObjectMaskAnnotation3DConfig defaults, REST parameter parsing,
and gRPC proto unmarshalling.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vizion3d.annotation.commands import ObjectMaskAnnotation3DCommand
from vizion3d.annotation.defaults import (
    DEFAULT_ANNOTATION_MODEL_FILENAME,
    DEFAULT_ANNOTATION_MODEL_URL,
)
from vizion3d.annotation.handlers import ObjectMaskAnnotation3DHandler
from vizion3d.annotation.models import ObjectMaskAnnotation3DConfig, ObjectMaskAnnotation3DResult

o3d = pytest.importorskip("open3d", reason="open3d required")


@pytest.fixture
def dummy_image_bytes():
    img = Image.new("RGB", (64, 48), color=(80, 120, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def small_point_cloud():
    pts = np.array([[0.0, 0.0, -2.0], [0.1, 0.1, -2.0]], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((2, 3)))
    return pcd


class TestConfigDefaults:
    def test_default_fx_fy(self):
        cfg = ObjectMaskAnnotation3DConfig()
        assert cfg.fx is None
        assert cfg.fy is None

    def test_default_cx_cy(self):
        cfg = ObjectMaskAnnotation3DConfig()
        assert cfg.cx is None
        assert cfg.cy is None

    def test_default_conf_threshold(self):
        assert ObjectMaskAnnotation3DConfig().conf_threshold == pytest.approx(0.25)

    def test_default_iou_threshold(self):
        assert ObjectMaskAnnotation3DConfig().iou_threshold == pytest.approx(0.45)

    def test_custom_intrinsics_accepted(self):
        cfg = ObjectMaskAnnotation3DConfig(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
        assert cfg.fx == pytest.approx(615.0)
        assert cfg.cx == pytest.approx(320.0)


class TestDefaultModelURL:
    def test_default_url_is_yolo11n_seg(self):
        assert "yolo26l-seg" in DEFAULT_ANNOTATION_MODEL_URL

    def test_default_filename_is_yolo26l_seg(self):
        assert DEFAULT_ANNOTATION_MODEL_FILENAME == "yolo26l-seg.pt"

    def test_command_default_backend_matches_default_url(self):
        pcd = o3d.geometry.PointCloud()
        cmd = ObjectMaskAnnotation3DCommand(point_cloud=pcd)
        assert cmd.model_backend == DEFAULT_ANNOTATION_MODEL_URL


class TestConfigPropagation:
    def test_custom_conf_threshold_reaches_yolo(self, dummy_image_bytes, small_point_cloud):
        calls = []

        def _fake(self_inner, model_path, image, cfg):
            calls.append(cfg.conf_threshold)
            return []

        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", _fake):
            ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                    advanced_config=ObjectMaskAnnotation3DConfig(conf_threshold=0.7),
                )
            )
        assert calls == [pytest.approx(0.7)]

    def test_custom_iou_threshold_reaches_yolo(self, dummy_image_bytes, small_point_cloud):
        calls = []

        def _fake(self_inner, model_path, image, cfg):
            calls.append(cfg.iou_threshold)
            return []

        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", _fake):
            ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                    advanced_config=ObjectMaskAnnotation3DConfig(iou_threshold=0.6),
                )
            )
        assert calls == [pytest.approx(0.6)]


class TestRESTConfigParsing:
    def test_rest_default_config_accepted(self, dummy_image_bytes, small_point_cloud):
        from fastapi.testclient import TestClient

        from vizion3d.lifting.utils import create_ply_binary
        from vizion3d.server.rest.app import app

        pts = np.asarray(small_point_cloud.points).astype(np.float32)
        cols = (np.asarray(small_point_cloud.colors) * 255).astype(np.uint8)
        ply_bytes = create_ply_binary(pts, cols)

        def _fake_run(*args, **kwargs):
            return ObjectMaskAnnotation3DResult(annotations=[], backend_used="/fake/model.pt")

        client = TestClient(app, raise_server_exceptions=True)
        with patch(
            "vizion3d.annotation.handlers.ObjectMaskAnnotation3DHandler.handle",
            side_effect=_fake_run,
        ):
            resp = client.post(
                "/annotation/object-mask-annotation-3d",
                files={"point_cloud_ply": ("cloud.ply", ply_bytes, "application/octet-stream")},
                data={"model_backend": "/fake/model.pt"},
            )
        assert resp.status_code == 200
        assert resp.json()["annotations"] == []


class TestGRPCConfigUnmarshalling:
    def test_grpc_default_config(self):
        from vizion3d.lifting.utils import create_ply_binary
        from vizion3d.proto import lifting_pb2
        from vizion3d.server.grpc.server import LiftingServiceServicer

        pts = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        cols = np.array([[128, 128, 128]], dtype=np.uint8)
        ply_bytes = create_ply_binary(pts, cols)

        request = lifting_pb2.ObjectMaskAnnotation3DRequest(
            point_cloud_ply=ply_bytes,
            model_backend="/fake/model.pt",
        )
        received = []

        def _fake(self_inner, cmd):
            received.append(cmd.advanced_config)
            return ObjectMaskAnnotation3DResult(annotations=[], backend_used="/fake/model.pt")

        ctx = MagicMock()
        with patch.object(ObjectMaskAnnotation3DHandler, "handle", _fake):
            LiftingServiceServicer().RunObjectMaskAnnotation3D(request, ctx)

        assert received[0].fx is None
        assert received[0].conf_threshold == pytest.approx(0.25)

    def test_grpc_custom_config_applied(self):
        from vizion3d.lifting.utils import create_ply_binary
        from vizion3d.proto import lifting_pb2
        from vizion3d.server.grpc.server import LiftingServiceServicer

        pts = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        ply_bytes = create_ply_binary(pts, np.array([[128, 128, 128]], dtype=np.uint8))

        proto_cfg = lifting_pb2.ObjectMaskAnnotation3DConfig(fx=615.0, conf_threshold=0.5)
        request = lifting_pb2.ObjectMaskAnnotation3DRequest(
            point_cloud_ply=ply_bytes,
            model_backend="/fake/model.pt",
            advanced_config=proto_cfg,
        )
        received = []

        def _fake(self_inner, cmd):
            received.append(cmd.advanced_config)
            return ObjectMaskAnnotation3DResult(annotations=[], backend_used="/fake/model.pt")

        ctx = MagicMock()
        with patch.object(ObjectMaskAnnotation3DHandler, "handle", _fake):
            LiftingServiceServicer().RunObjectMaskAnnotation3D(request, ctx)

        assert received[0].fx == pytest.approx(615.0)
        assert received[0].conf_threshold == pytest.approx(0.5)
        assert received[0].cy is None  # unset → None, handler resolves from image
