import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("open3d", reason="open3d required — run: uv python pin 3.12 && uv sync")

from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_URL  # noqa: E402
from vizion3d.server.rest.app import app, create_app, run  # noqa: E402
from vizion3d.stereo.defaults import DEFAULT_STEREO_MODEL_URL  # noqa: E402

client = TestClient(app)


@pytest.fixture
def image_file():
    img = Image.new("RGB", (50, 50), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def stereo_files():
    img = Image.new("RGB", (50, 50), color="green")
    left = io.BytesIO()
    right = io.BytesIO()
    img.save(left, format="PNG")
    img.save(right, format="PNG")
    left.seek(0)
    right.seek(0)
    return left, right


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


def _fake_stereo_result():
    result = _fake_result()
    result.disparity_map = [[4.0, 5.0], [6.0, 7.0]]
    result.backend_used = "/fake/stereo.pth"
    return result


def _uvicorn_app_routes(uvicorn_run) -> set[str]:
    app_arg = uvicorn_run.call_args.args[0]
    return {route.path for route in app_arg.routes}


def test_depth_estimation_returns_200(image_file):
    with patch("vizion3d.server.rest.depth_estimation.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    assert response.status_code == 200


def test_depth_estimation_response_has_expected_keys(image_file):
    with patch("vizion3d.server.rest.depth_estimation.DepthEstimation") as mock_cls:
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
    with patch("vizion3d.server.rest.depth_estimation.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    data = response.json()
    assert isinstance(data["depth_map"], list)
    assert all(isinstance(row, list) for row in data["depth_map"])


def test_depth_estimation_max_depth_gte_min_depth(image_file):
    with patch("vizion3d.server.rest.depth_estimation.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    data = response.json()
    assert data["max_depth"] >= data["min_depth"]


def test_depth_estimation_optional_outputs_null_by_default(image_file):
    with patch("vizion3d.server.rest.depth_estimation.DepthEstimation") as mock_cls:
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
    with patch("vizion3d.server.rest.depth_estimation.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    data = response.json()
    assert data["backend_used"] == "/fake/model.pth"


def test_depth_estimation_passes_form_fields_to_command(image_file):
    with patch("vizion3d.server.rest.depth_estimation.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
            data={"model_backend": "/my/model.pth", "return_depth_image": "false"},
        )
    called_cmd = mock_cls.return_value.run.call_args[0][0]
    assert called_cmd.model_backend == "/my/model.pth"
    assert called_cmd.return_depth_image is False


def test_depth_estimation_uses_default_backend_when_omitted(image_file):
    with patch("vizion3d.server.rest.depth_estimation.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        client.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    called_cmd = mock_cls.return_value.run.call_args[0][0]
    assert called_cmd.model_backend == DEFAULT_DEPTH_MODEL_URL


def test_depth_estimation_missing_image_returns_422():
    response = client.post("/lifting/depth-estimation")
    assert response.status_code == 422


def test_stereo_depth_returns_200(stereo_files):
    left, right = stereo_files
    with patch("vizion3d.server.rest.stereo_depth.StereoDepth") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_stereo_result()
        response = client.post(
            "/lifting/stereo-depth",
            files={
                "left_image": ("left.png", left, "image/png"),
                "right_image": ("right.png", right, "image/png"),
            },
        )
    assert response.status_code == 200


def test_stereo_depth_response_has_expected_keys(stereo_files):
    left, right = stereo_files
    with patch("vizion3d.server.rest.stereo_depth.StereoDepth") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_stereo_result()
        response = client.post(
            "/lifting/stereo-depth",
            files={
                "left_image": ("left.png", left, "image/png"),
                "right_image": ("right.png", right, "image/png"),
            },
        )
    data = response.json()
    assert "depth_map" in data
    assert "disparity_map" in data
    assert "min_depth" in data
    assert "max_depth" in data
    assert "backend_used" in data


def test_stereo_depth_uses_default_backend_when_omitted(stereo_files):
    left, right = stereo_files
    with patch("vizion3d.server.rest.stereo_depth.StereoDepth") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_stereo_result()
        client.post(
            "/lifting/stereo-depth",
            files={
                "left_image": ("left.png", left, "image/png"),
                "right_image": ("right.png", right, "image/png"),
            },
        )
    called_cmd = mock_cls.return_value.run.call_args[0][0]
    assert called_cmd.model_backend == DEFAULT_STEREO_MODEL_URL


def test_stereo_depth_passes_form_fields_to_command(stereo_files):
    left, right = stereo_files
    with patch("vizion3d.server.rest.stereo_depth.StereoDepth") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_stereo_result()
        client.post(
            "/lifting/stereo-depth",
            files={
                "left_image": ("left.png", left, "image/png"),
                "right_image": ("right.png", right, "image/png"),
            },
            data={
                "model_backend": "/my/stereo.pth",
                "return_depth_image": "true",
                "focal_length": "1733.74",
                "baseline": "536.62",
                "scale_factor": "0.5",
            },
        )
    called_cmd = mock_cls.return_value.run.call_args[0][0]
    assert called_cmd.model_backend == "/my/stereo.pth"
    assert called_cmd.return_depth_image is True
    assert called_cmd.advanced_config.focal_length == pytest.approx(1733.74)
    assert called_cmd.advanced_config.baseline == pytest.approx(536.62)
    assert called_cmd.advanced_config.scale_factor == pytest.approx(0.5)


def test_create_app_depth_only_disables_stereo_endpoint(stereo_files, image_file):
    depth_only = TestClient(create_app(enable_depth_estimation=True, enable_stereo_depth=False))
    response = depth_only.post(
        "/lifting/stereo-depth",
        files={
            "left_image": ("left.png", stereo_files[0], "image/png"),
            "right_image": ("right.png", stereo_files[1], "image/png"),
        },
    )
    assert response.status_code == 404

    with patch("vizion3d.server.rest.depth_estimation.DepthEstimation") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_result()
        response = depth_only.post(
            "/lifting/depth-estimation",
            files={"image": ("test.png", image_file, "image/png")},
        )
    assert response.status_code == 200


def test_create_app_stereo_only_disables_depth_endpoint(stereo_files, image_file):
    stereo_only = TestClient(create_app(enable_depth_estimation=False, enable_stereo_depth=True))
    response = stereo_only.post(
        "/lifting/depth-estimation",
        files={"image": ("test.png", image_file, "image/png")},
    )
    assert response.status_code == 404

    left, right = stereo_files
    with patch("vizion3d.server.rest.stereo_depth.StereoDepth") as mock_cls:
        mock_cls.return_value.run.return_value = _fake_stereo_result()
        response = stereo_only.post(
            "/lifting/stereo-depth",
            files={
                "left_image": ("left.png", left, "image/png"),
                "right_image": ("right.png", right, "image/png"),
            },
        )
    assert response.status_code == 200


def test_run_preloads_enabled_feature_model_paths():
    with (
        patch("vizion3d.server.rest.app.depth_estimation.configure_model") as depth_cfg,
        patch("vizion3d.server.rest.app.stereo_depth.configure_model") as stereo_cfg,
        patch("vizion3d.server.rest.app.uvicorn.run") as uvicorn_run,
    ):
        run(
            [
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
                "--depth_model",
                "/models/depth.pth",
                "--stereo_model",
                "/models/stereo.pth",
            ]
        )

    depth_cfg.assert_called_once_with("/models/depth.pth")
    stereo_cfg.assert_called_once_with("/models/stereo.pth")
    uvicorn_run.assert_called_once()
    assert uvicorn_run.call_args.kwargs["host"] == "127.0.0.1"
    assert uvicorn_run.call_args.kwargs["port"] == 9000
    assert "/lifting/depth-estimation" in _uvicorn_app_routes(uvicorn_run)
    assert "/lifting/stereo-depth" in _uvicorn_app_routes(uvicorn_run)


def test_run_model_path_enables_and_preloads_feature():
    with (
        patch("vizion3d.server.rest.app.depth_estimation.configure_model") as depth_cfg,
        patch("vizion3d.server.rest.app.stereo_depth.configure_model") as stereo_cfg,
        patch("vizion3d.server.rest.app.uvicorn.run") as uvicorn_run,
    ):
        run(
            [
                "--depth_model",
                "/models/depth.pth",
            ]
        )

    depth_cfg.assert_called_once_with("/models/depth.pth")
    stereo_cfg.assert_not_called()
    assert "/lifting/depth-estimation" in _uvicorn_app_routes(uvicorn_run)
    assert "/lifting/stereo-depth" not in _uvicorn_app_routes(uvicorn_run)


def test_run_only_selected_feature_does_not_enable_other_endpoint():
    with (
        patch("vizion3d.server.rest.app.depth_estimation.configure_model") as depth_cfg,
        patch("vizion3d.server.rest.app.stereo_depth.configure_model") as stereo_cfg,
        patch("vizion3d.server.rest.app.uvicorn.run") as uvicorn_run,
    ):
        run(["--stereo_depth", "--stereo_model", "/models/stereo.pth"])

    depth_cfg.assert_not_called()
    stereo_cfg.assert_called_once_with("/models/stereo.pth")
    assert "/lifting/depth-estimation" not in _uvicorn_app_routes(uvicorn_run)
    assert "/lifting/stereo-depth" in _uvicorn_app_routes(uvicorn_run)
