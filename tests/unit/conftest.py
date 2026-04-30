import io

import pytest
from PIL import Image


@pytest.fixture
def dummy_image_bytes():
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
