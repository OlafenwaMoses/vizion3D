"""
Unit tests for the vendored SegFormer-B4 backbone.

These run on a randomly-initialised model (no checkpoint needed) and verify the
architecture's output shapes, arbitrary-input-size support, and the helper
utilities.  Weight loading is covered by the integration tests.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from vizion3d.annotation import segformer as S


class TestArchitectureShapes:
    def test_forward_output_is_quarter_resolution(self):
        model = S.SegformerForSemanticSegmentation().eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 128, 128))
        # decode head emits logits at the stride-4 feature size
        assert out.shape == (1, S.NUM_CLASSES, 32, 32)

    def test_accepts_non_square_input(self):
        model = S.SegformerForSemanticSegmentation().eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 160, 96))
        assert out.shape == (1, S.NUM_CLASSES, 40, 24)


class TestInferSemantic:
    def test_returns_class_map_at_original_size(self):
        model = S.SegformerForSemanticSegmentation().eval()
        img = Image.new("RGB", (50, 30), color=(10, 20, 30))  # (W, H)
        seg = S.infer_semantic(model, img, torch.device("cpu"), inference_size=32)
        assert seg.shape == (30, 50)
        assert seg.dtype == np.int32
        assert seg.min() >= 0 and seg.max() < S.NUM_CLASSES

    def test_native_resolution_when_inference_size_zero(self):
        model = S.SegformerForSemanticSegmentation().eval()
        img = Image.new("RGB", (64, 48))
        seg = S.infer_semantic(model, img, torch.device("cpu"), inference_size=0)
        assert seg.shape == (48, 64)


class TestUtilities:
    def test_palette_shape_and_dtype(self):
        assert S.ADE20K_PALETTE.shape == (S.NUM_CLASSES, 3)
        assert S.ADE20K_PALETTE.dtype == np.uint8

    def test_colorize_maps_to_rgb(self):
        seg = np.array([[0, 1], [2, 3]], dtype=np.int32)
        col = S.colorize(seg)
        assert col.shape == (2, 2, 3)
        assert np.array_equal(col[0, 0], S.ADE20K_PALETTE[0])

    def test_class_list_has_150_entries(self):
        assert len(S.ADE20K_CLASSES) == S.NUM_CLASSES

    def test_coco_aligned_names_applied(self):
        # the 7 unambiguous renames use the COCO name, not the ADE name
        for coco_name in ["couch", "dining table", "potted plant", "laptop", "tv",
                          "motorcycle", "wine glass"]:
            assert coco_name in S.ADE20K_CLASSES
        for ade_name in ["sofa", "television receiver", "minibike", "computer"]:
            assert ade_name not in S.ADE20K_CLASSES
        # ambiguous many-to-one cases keep the ADE name
        assert "animal" in S.ADE20K_CLASSES
        assert "pot" in S.ADE20K_CLASSES


class TestPreprocess:
    def test_resizes_shorter_edge(self):
        img = Image.new("RGB", (200, 100))  # shorter edge = 100
        tensor, (w, h) = S._preprocess(img, inference_size=50)
        assert (w, h) == (200, 100)  # original size reported
        # shorter edge scaled to 50 → 100*0.5; longer edge 200*0.5 = 100
        assert tensor.shape == (1, 3, 50, 100)
