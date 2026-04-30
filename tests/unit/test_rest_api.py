import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("open3d", reason="open3d required — run: uv python pin 3.12 && uv sync")

from vizion3d.server.rest.app import app  # noqa: E402

client = TestClient(app)


@pytest.fixture
def image_file():
    img = Image.new("RGB", (50, 50), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


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


def test_depth_estimation_returns_200(image_file):
    with patch("vizion3d.server.rest.app.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    assert response.status_code == 200


def test_depth_estimation_response_has_expected_keys(image_file):
    with patch("vizion3d.server.rest.app.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    data = response.json()
    assert "depth_map" in data
    assert "min_depth" in data
    assert "max_depth" in data
    assert "backend_used" in data


def test_depth_estimation_depth_map_is_nested_list(image_file):
    with patch("vizion3d.server.rest.app.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    data = response.json()
    assert isinstance(data["depth_map"], list)
    assert all(isinstance(row, list) for row in data["depth_map"])


def test_depth_estimation_max_depth_gte_min_depth(image_file):
    with patch("vizion3d.server.rest.app.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    data = response.json()
    assert data["max_depth"] >= data["min_depth"]


def test_depth_estimation_optional_outputs_null_by_default(image_file):
    with patch("vizion3d.server.rest.app.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    data = response.json()
    assert data["depth_image"] is None
    assert data["point_cloud_ply"] is None
    assert data["mesh_ply"] is None


def test_depth_estimation_backend_used_is_returned(image_file):
    with patch("vizion3d.server.rest.app.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    data = response.json()
    assert data["backend_used"] == "/fake/model.pth"


def test_depth_estimation_passes_form_fields_to_command(image_file):
    with patch("vizion3d.server.rest.app.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
            data={"model_backend": "/my/model.pth", "return_depth_image": "false"},
        )
    called_cmd = mock_cls.return_value.run.call_args[0][0]
    assert called_cmd.model_backend == "/my/model.pth"
    assert called_cmd.return_depth_image is False


def test_depth_estimation_missing_image_returns_422():
    response = client.post("/lifting/depth-estimation")
    assert response.status_code == 422
