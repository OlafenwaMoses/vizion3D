"""Unit tests for SceneMaskAnnotation3DConfig defaults and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vizion3d.annotation.models import SceneMaskAnnotation3DConfig


class TestDefaults:
    def test_intrinsics_default_to_none(self):
        cfg = SceneMaskAnnotation3DConfig()
        assert cfg.fx is None
        assert cfg.fy is None
        assert cfg.cx is None
        assert cfg.cy is None

    def test_default_inference_size(self):
        assert SceneMaskAnnotation3DConfig().inference_size == 512


class TestOverrides:
    def test_set_intrinsics(self):
        cfg = SceneMaskAnnotation3DConfig(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
        assert cfg.fx == 525.0
        assert cfg.cy == 239.5

    def test_set_inference_size(self):
        assert SceneMaskAnnotation3DConfig(inference_size=0).inference_size == 0
        assert SceneMaskAnnotation3DConfig(inference_size=640).inference_size == 640

    def test_model_copy_preserves_inference_size(self):
        cfg = SceneMaskAnnotation3DConfig(inference_size=640)
        updated = cfg.model_copy(update={"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 24.0})
        assert updated.inference_size == 640
        assert updated.fx == 100.0


class TestValidation:
    def test_inference_size_must_be_int(self):
        with pytest.raises(ValidationError):
            SceneMaskAnnotation3DConfig(inference_size="big")
