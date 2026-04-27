import numpy as np
import pytest
from PIL import Image
from transformers import pipeline


@pytest.fixture(scope="module")
def hf_pipeline():
    return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")


@pytest.fixture(scope="module")
def hf_result(hf_pipeline):
    image = Image.new("RGB", (100, 100), color="red")
    return hf_pipeline(image), image


def test_hf_pipeline_returns_expected_keys(hf_result):
    result, _ = hf_result
    assert "depth" in result
    assert "predicted_depth" in result


def test_hf_pipeline_depth_is_pil_image(hf_result):
    result, _ = hf_result
    assert isinstance(result["depth"], Image.Image)


def test_hf_pipeline_predicted_depth_is_tensor(hf_result):
    result, _ = hf_result
    import torch

    assert isinstance(result["predicted_depth"], torch.Tensor)


def test_hf_pipeline_predicted_depth_shape(hf_result):
    result, image = hf_result
    tensor = result["predicted_depth"].squeeze()
    # The pipeline resizes predicted_depth to match input image dimensions
    assert tensor.ndim == 2
    assert tensor.shape == (image.size[1], image.size[0])


def test_hf_pipeline_depth_values_are_finite(hf_result):
    result, _ = hf_result
    import torch

    tensor = result["predicted_depth"]
    assert torch.isfinite(tensor).all()
    assert tensor.min().item() >= 0
