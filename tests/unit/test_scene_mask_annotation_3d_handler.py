"""
Unit tests for SceneMaskAnnotation3DHandler internals.

All tests mock ``_run_segformer`` so no SegFormer checkpoint is needed on disk.
The mock returns a synthetic HxW class-id map.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vizion3d.annotation.commands import SceneMaskAnnotation3DCommand
from vizion3d.annotation._geometry import _extract_cloud_arrays, _render_front_view
from vizion3d.annotation.models import SceneMaskAnnotation3DConfig
from vizion3d.annotation.scene_handlers import SceneMaskAnnotation3DHandler
from vizion3d.annotation.segformer import ADE20K_CLASSES, ADE20K_PALETTE

o3d = pytest.importorskip("open3d", reason="open3d required")


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_image_bytes():
    img = Image.new("RGB", (64, 48), color=(80, 120, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def small_point_cloud():
    """4×4 OpenGL camera-space grid of points in front of the default camera."""
    pts, cols = [], []
    for v in range(4):
        for u in range(4):
            pts.append([(u - 1.5) * 0.1, (1.5 - v) * 0.1, -2.0])
            cols.append([0.5, 0.5, 0.5])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.array(cols, dtype=np.float64))
    return pcd


def _cfg():
    return SceneMaskAnnotation3DConfig(fx=100.0, fy=100.0, cx=32.0, cy=24.0)


def _seg_single(class_id=0, h=48, w=64):
    return np.full((h, w), class_id, dtype=np.int32)


def _seg_split(top_id=0, bottom_id=3, h=48, w=64):
    seg = np.full((h, w), top_id, dtype=np.int32)
    seg[h // 2 :, :] = bottom_id
    return seg


# ── Device selection ──────────────────────────────────────────────────────────


class TestDeviceSelection:
    def test_returns_cuda_when_available(self):
        t = MagicMock()
        t.cuda.is_available.return_value = True
        assert SceneMaskAnnotation3DHandler._torch_device(t) == "cuda"

    def test_returns_mps_when_cuda_unavailable(self):
        t = MagicMock()
        t.cuda.is_available.return_value = False
        t.backends.mps.is_available.return_value = True
        assert SceneMaskAnnotation3DHandler._torch_device(t) == "mps"

    def test_returns_cpu_as_fallback(self):
        t = MagicMock()
        t.cuda.is_available.return_value = False
        t.backends.mps.is_available.return_value = False
        assert SceneMaskAnnotation3DHandler._torch_device(t) == "cpu"


# ── Per-class grouping ──────────────────────────────────────────────────────────


class TestPerClassGrouping:
    def test_single_class_groups_all_points(self, dummy_image_bytes, small_point_cloud):
        seg = _seg_single(class_id=0)
        with patch.object(SceneMaskAnnotation3DHandler, "_run_segformer", return_value=seg):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.bin",
                    advanced_config=_cfg(),
                )
            )
        assert len(result.annotations) == 1
        ann = result.annotations[0]
        assert ann.label == ADE20K_CLASSES[0]
        assert ann.class_id == 0
        assert len(ann.point_indices) == 16  # all grid points land in-bounds

    def test_two_classes_yield_two_annotations(self, dummy_image_bytes, small_point_cloud):
        seg = _seg_split(top_id=0, bottom_id=3)
        with patch.object(SceneMaskAnnotation3DHandler, "_run_segformer", return_value=seg):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.bin",
                    advanced_config=_cfg(),
                )
            )
        labels = {a.label for a in result.annotations}
        assert labels == {ADE20K_CLASSES[0], ADE20K_CLASSES[3]}
        # every in-bounds point is assigned to exactly one class
        total = sum(len(a.point_indices) for a in result.annotations)
        assert total == 16

    def test_annotations_sorted_by_pixel_count_desc(self, dummy_image_bytes, small_point_cloud):
        seg = np.full((48, 64), 0, dtype=np.int32)
        seg[:5, :] = 3  # small class 3, large class 0
        with patch.object(SceneMaskAnnotation3DHandler, "_run_segformer", return_value=seg):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.bin",
                    advanced_config=_cfg(),
                )
            )
        counts = [a.pixel_count for a in result.annotations]
        assert counts == sorted(counts, reverse=True)
        assert result.annotations[0].class_id == 0


# ── Output structure ────────────────────────────────────────────────────────────


class TestOutputStructure:
    def test_annotation_fields(self, dummy_image_bytes, small_point_cloud):
        seg = _seg_single(class_id=23)  # couch
        with patch.object(SceneMaskAnnotation3DHandler, "_run_segformer", return_value=seg):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.bin",
                    advanced_config=_cfg(),
                )
            )
        ann = result.annotations[0]
        assert ann.label == "couch"
        assert ann.mask_2d.dtype == bool
        assert ann.mask_2d.shape == (48, 64)
        assert ann.pixel_count == 48 * 64
        assert len(ann.bbox_2d) == 4
        assert len(ann.point_coords) == len(ann.point_indices)
        assert all(len(c) == 3 for c in ann.point_coords)

    def test_bbox_spans_class_region(self, dummy_image_bytes, small_point_cloud):
        seg = _seg_single(class_id=0)
        with patch.object(SceneMaskAnnotation3DHandler, "_run_segformer", return_value=seg):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.bin",
                    advanced_config=_cfg(),
                )
            )
        assert result.annotations[0].bbox_2d == [0.0, 0.0, 63.0, 47.0]


# ── Optional outputs ────────────────────────────────────────────────────────────


class TestOptionalOutputs:
    def test_region_cloud_absent_by_default(self, dummy_image_bytes, small_point_cloud):
        seg = _seg_single(class_id=0)
        with patch.object(SceneMaskAnnotation3DHandler, "_run_segformer", return_value=seg):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.bin",
                    advanced_config=_cfg(),
                )
            )
        assert result.annotations[0].region_cloud is None

    def test_region_cloud_returned_when_requested(self, dummy_image_bytes, small_point_cloud):
        seg = _seg_single(class_id=0)
        with patch.object(SceneMaskAnnotation3DHandler, "_run_segformer", return_value=seg):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.bin",
                    return_region_clouds=True,
                    advanced_config=_cfg(),
                )
            )
        ann = result.annotations[0]
        assert isinstance(ann.region_cloud, o3d.geometry.PointCloud)
        assert len(np.asarray(ann.region_cloud.points)) == len(ann.point_indices)

    def test_region_cloud_preserves_original_colours(self, dummy_image_bytes, small_point_cloud):
        seg = _seg_single(class_id=0)
        with patch.object(SceneMaskAnnotation3DHandler, "_run_segformer", return_value=seg):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.bin",
                    return_region_clouds=True,
                    return_annotated_cloud=True,
                    advanced_config=_cfg(),
                )
            )
        ann = result.annotations[0]
        orig = np.asarray(small_point_cloud.colors)[ann.point_indices]
        got = np.asarray(ann.region_cloud.colors)
        assert np.allclose(orig, got)

    def test_annotated_cloud_recoloured_by_palette(self, dummy_image_bytes, small_point_cloud):
        seg = _seg_single(class_id=0)
        with patch.object(SceneMaskAnnotation3DHandler, "_run_segformer", return_value=seg):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.bin",
                    return_annotated_cloud=True,
                    advanced_config=_cfg(),
                )
            )
        assert result.annotated_cloud is not None
        n_in = len(np.asarray(small_point_cloud.points))
        cols = np.asarray(result.annotated_cloud.colors)
        assert len(cols) == n_in
        expected = ADE20K_PALETTE[0] / 255.0
        assert np.allclose(cols[result.annotations[0].point_indices], expected)


# ── No-image front-view path ────────────────────────────────────────────────────


class TestNoImageInput:
    def test_front_view_synthesis_uses_intrinsic_viewport(self, small_point_cloud):
        pts, cols = _extract_cloud_arrays(small_point_cloud)
        cfg = _cfg()
        image = _render_front_view(pts, cols, cfg)
        assert image.size == (65, 49)

    def test_handle_without_image_runs(self, small_point_cloud):
        seen_sizes = []

        def _seg_for_image(model_path, image, cfg):
            w, h = image.size
            seen_sizes.append((w, h))
            return np.zeros((h, w), dtype=np.int32)

        with patch.object(
            SceneMaskAnnotation3DHandler, "_run_segformer", side_effect=_seg_for_image
        ):
            result = SceneMaskAnnotation3DHandler().handle(
                SceneMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=None,
                    model_backend="/fake/model.bin",
                    advanced_config=_cfg(),
                )
            )
        assert isinstance(result.annotations, list)
        assert len(result.annotations) >= 1
        assert seen_sizes == [(65, 49)]
