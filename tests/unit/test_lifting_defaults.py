import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from vizion3d.lifting import defaults
from vizion3d.lifting.defaults import (
    DEFAULT_DEPTH_MODEL_BACKEND,
    DEFAULT_DEPTH_MODEL_FILENAME,
    DEFAULT_DEPTH_MODEL_URL,
    resolve_model_backend,
)
from vizion3d.lifting.handlers import DepthEstimationHandler


def test_default_backend_downloads_to_cache(tmp_path, monkeypatch):
    def fake_download(url, cache_dir=None):
        assert url == DEFAULT_DEPTH_MODEL_URL
        return (cache_dir or tmp_path) / DEFAULT_DEPTH_MODEL_FILENAME

    monkeypatch.setattr(defaults, "download_model", fake_download)

    resolved = resolve_model_backend(DEFAULT_DEPTH_MODEL_BACKEND, cache_dir=tmp_path)

    assert resolved == str(tmp_path / DEFAULT_DEPTH_MODEL_FILENAME)


def test_explicit_filepath_is_left_untouched():
    assert resolve_model_backend("/models/custom.pth") == "/models/custom.pth"


def test_loaded_checkpoint_cache_is_shared_across_handler_instances():
    original_cache = DepthEstimationHandler._depth_anything_models
    cache_key = "/models/custom.pth"
    cached_model = (object(), object())

    try:
        DepthEstimationHandler._depth_anything_models = {cache_key: cached_model}

        assert (
            DepthEstimationHandler()._load_depth_anything_checkpoint(cache_key)
            is cached_model
        )
        assert (
            DepthEstimationHandler()._load_depth_anything_checkpoint(cache_key)
            is cached_model
        )
    finally:
        DepthEstimationHandler._depth_anything_models = original_cache


def test_pth_path_is_detected_as_checkpoint():
    assert DepthEstimationHandler._is_depth_anything_checkpoint("/models/custom.pth") is True
    assert DepthEstimationHandler._is_depth_anything_checkpoint("/models/custom.pt") is True


def test_non_pth_path_is_not_detected_as_checkpoint():
    assert DepthEstimationHandler._is_depth_anything_checkpoint("depth-anything/model-hf") is False
    assert DepthEstimationHandler._is_depth_anything_checkpoint("/models/model.bin") is False


def test_local_pth_model_backend_runs_checkpoint_path(tmp_path):
    img = Image.new("RGB", (50, 50), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    fake_depth = np.ones((50, 50), dtype=float) * 2.5
    fake_model_path = str(tmp_path / "model.pth")

    with patch.object(
        DepthEstimationHandler,
        "_run_depth_anything_checkpoint",
        return_value=fake_depth,
    ) as mock_checkpoint, patch.object(
        DepthEstimationHandler, "_load_hugging_face_pipeline"
    ) as mock_hf:
        handler = DepthEstimationHandler()
        from vizion3d.lifting.commands import DepthEstimationCommand

        result = handler.handle(
            DepthEstimationCommand(image_input=image_bytes, model_backend=fake_model_path)
        )

    mock_checkpoint.assert_called_once()
    mock_hf.assert_not_called()
    assert result.backend_used == fake_model_path
    assert result.min_depth == pytest.approx(2.5)
    assert result.max_depth == pytest.approx(2.5)


def test_local_pth_model_depth_map_matches_fake_depth(tmp_path):
    img = Image.new("RGB", (4, 3), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    fake_depth = np.array([[1.0, 2.0, 3.0, 4.0]] * 3, dtype=float)
    fake_model_path = str(tmp_path / "model.pth")

    with patch.object(
        DepthEstimationHandler,
        "_run_depth_anything_checkpoint",
        return_value=fake_depth,
    ):
        handler = DepthEstimationHandler()
        from vizion3d.lifting.commands import DepthEstimationCommand

        result = handler.handle(
            DepthEstimationCommand(image_input=image_bytes, model_backend=fake_model_path)
        )

    assert result.depth_map == fake_depth.tolist()


def test_url_model_backend_is_downloaded(tmp_path, monkeypatch):
    fake_url = "https://example.com/weights/model.pth"
    expected_path = tmp_path / "model.pth"

    def fake_download(url, cache_dir=None):
        assert url == fake_url
        return expected_path

    monkeypatch.setattr(defaults, "download_model", fake_download)

    resolved = resolve_model_backend(fake_url, cache_dir=tmp_path)

    assert resolved == str(expected_path)
