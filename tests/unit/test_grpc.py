import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

pytest.importorskip("open3d", reason="open3d required — run: uv python pin 3.12 && uv sync")

from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_URL  # noqa: E402
from vizion3d.proto import lifting_pb2  # noqa: E402
from vizion3d.server.grpc.server import LiftingServiceServicer  # noqa: E402


@pytest.fixture(scope="module")
def image_bytes():
    img = Image.new("RGB", (50, 50), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def servicer():
    return LiftingServiceServicer()


@pytest.fixture
def mock_context():
    return MagicMock()


def _fake_result(depth_map=None):
    result = MagicMock()
    result.depth_map = depth_map or [[1.0, 2.0], [3.0, 4.0]]
    result.min_depth = 1.0
    result.max_depth = 4.0
    result.backend_used = "/fake/model.pth"
    result.depth_image = None
    result.point_cloud = None
    result.mesh = None
    return result


def test_grpc_run_returns_response_type(servicer, mock_context, image_bytes):
    request = lifting_pb2.DepthEstimationRequest(image_bytes=image_bytes)
    with patch("vizion3d.server.grpc.server.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = servicer.RunDepthEstimation(request, mock_context)
    assert isinstance(response, lifting_pb2.DepthEstimationResponse)


def test_grpc_response_contains_depth_map_rows(servicer, mock_context, image_bytes):
    request = lifting_pb2.DepthEstimationRequest(image_bytes=image_bytes)
    with patch("vizion3d.server.grpc.server.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = servicer.RunDepthEstimation(request, mock_context)
    assert len(response.depth_map) == 2
    assert list(response.depth_map[0].values) == pytest.approx([1.0, 2.0])
    assert list(response.depth_map[1].values) == pytest.approx([3.0, 4.0])


def test_grpc_response_min_max_depth(servicer, mock_context, image_bytes):
    request = lifting_pb2.DepthEstimationRequest(image_bytes=image_bytes)
    with patch("vizion3d.server.grpc.server.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = servicer.RunDepthEstimation(request, mock_context)
    assert response.min_depth == pytest.approx(1.0)
    assert response.max_depth == pytest.approx(4.0)
    assert response.max_depth >= response.min_depth


def test_grpc_response_backend_used(servicer, mock_context, image_bytes):
    request = lifting_pb2.DepthEstimationRequest(image_bytes=image_bytes)
    with patch("vizion3d.server.grpc.server.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = servicer.RunDepthEstimation(request, mock_context)
    assert response.backend_used == "/fake/model.pth"


def test_grpc_optional_fields_empty_when_not_requested(servicer, mock_context, image_bytes):
    request = lifting_pb2.DepthEstimationRequest(image_bytes=image_bytes)
    with patch("vizion3d.server.grpc.server.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = servicer.RunDepthEstimation(request, mock_context)
    assert response.depth_image == b""
    assert response.point_cloud_ply == b""
    assert response.mesh_ply == b""


def test_grpc_uses_default_backend_when_model_backend_is_empty(servicer, mock_context, image_bytes):
    request = lifting_pb2.DepthEstimationRequest(image_bytes=image_bytes)
    with patch("vizion3d.server.grpc.server.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        servicer.RunDepthEstimation(request, mock_context)
    called_cmd = mock_cls.return_value.run.call_args[0][0]
    assert called_cmd.model_backend == DEFAULT_DEPTH_MODEL_URL


def test_grpc_forwards_custom_model_backend(servicer, mock_context, image_bytes):
    request = lifting_pb2.DepthEstimationRequest(
        image_bytes=image_bytes, model_backend="/custom/model.pth"
    )
    with patch("vizion3d.server.grpc.server.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        servicer.RunDepthEstimation(request, mock_context)
    called_cmd = mock_cls.return_value.run.call_args[0][0]
    assert called_cmd.model_backend == "/custom/model.pth"


def test_grpc_forwards_return_flags(servicer, mock_context, image_bytes):
    request = lifting_pb2.DepthEstimationRequest(
        image_bytes=image_bytes,
        return_depth_image=True,
        return_point_cloud=True,
        return_mesh=True,
    )
    with patch("vizion3d.server.grpc.server.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        servicer.RunDepthEstimation(request, mock_context)
    called_cmd = mock_cls.return_value.run.call_args[0][0]
    assert called_cmd.return_depth_image is True
    assert called_cmd.return_point_cloud is True
    assert called_cmd.return_mesh is True
