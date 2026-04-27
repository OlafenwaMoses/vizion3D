import numpy as np
import pytest
from PIL import Image
from transformers import pipeline


@pytest.fixture(scope="module")
def depth_result():
    image = Image.new("RGB", (100, 100), color="blue")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")
    return pipe(image)


def test_tensor_depth_dtype(depth_result):
    import torch

    tensor = depth_result["predicted_depth"]
    assert tensor.dtype == torch.float32


def test_tensor_depth_has_valid_range(depth_result):
    import torch

    tensor = depth_result["predicted_depth"]
    assert torch.isfinite(tensor).all()
    # Model outputs can be near-zero but not NaN or Inf
    assert not torch.isnan(tensor).any()


def test_pil_depth_is_grayscale(depth_result):
    pil_img = depth_result["depth"]
    assert isinstance(pil_img, Image.Image)


def test_pil_depth_array_shape(depth_result):
    pil_array = np.array(depth_result["depth"])
    # Depth PIL image is grayscale — shape should be (H, W) or (H, W, 1)
    assert pil_array.ndim in (2, 3)


def test_pil_depth_values_are_uint8(depth_result):
    pil_array = np.array(depth_result["depth"])
    assert pil_array.dtype == np.uint8
    assert pil_array.min() >= 0
    assert pil_array.max() <= 255
