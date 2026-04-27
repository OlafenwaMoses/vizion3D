from vision3d.lifting import defaults
from vision3d.lifting.defaults import (
    DEFAULT_DEPTH_MODEL_BACKEND,
    DEFAULT_DEPTH_MODEL_FILENAME,
    DEFAULT_DEPTH_MODEL_URL,
    resolve_model_backend,
)
from vision3d.lifting.handlers import DepthEstimationHandler


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
