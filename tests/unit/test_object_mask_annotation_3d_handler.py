"""
Unit tests for ObjectMaskAnnotation3DHandler internals.

All tests mock _run_yolo so no YOLO checkpoint is needed on disk.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vizion3d.annotation.commands import ObjectMaskAnnotation3DCommand
from vizion3d.annotation.handlers import (
    ObjectMaskAnnotation3DHandler,
    _backproject,
    _extract_cloud_arrays,
    _render_front_view,
)
from vizion3d.annotation.models import ObjectMaskAnnotation3DConfig

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
    """4×4 grid of points in front of the default camera."""
    pts, cols = [], []
    for v in range(4):
        for u in range(4):
            pts.append([(u - 1.5) * 0.1, (v - 1.5) * 0.1, 2.0])
            cols.append([0.5, 0.5, 0.5])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.array(cols, dtype=np.float64))
    return pcd


@pytest.fixture
def full_mask_48x64():
    return np.ones((48, 64), dtype=bool)


@pytest.fixture
def empty_mask_48x64():
    return np.zeros((48, 64), dtype=bool)


def _make_detection(mask, label="chair", class_id=56, conf=0.85):
    return (label, class_id, conf, [0.0, 0.0, 64.0, 48.0], mask)


# ── Device selection ──────────────────────────────────────────────────────────


class TestDeviceSelection:
    def test_returns_cuda_when_available(self):
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        assert ObjectMaskAnnotation3DHandler._torch_device(torch_mock) == "cuda"

    def test_returns_mps_when_cuda_unavailable(self):
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        torch_mock.backends.mps.is_available.return_value = True
        assert ObjectMaskAnnotation3DHandler._torch_device(torch_mock) == "mps"

    def test_returns_cpu_as_fallback(self):
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        torch_mock.backends.mps.is_available.return_value = False
        assert ObjectMaskAnnotation3DHandler._torch_device(torch_mock) == "cpu"


# ── Back-projection helper ────────────────────────────────────────────────────


class TestBackprojection:
    def test_points_in_front_project_into_image(self, small_point_cloud):
        pts, _ = _extract_cloud_arrays(small_point_cloud)
        cfg = ObjectMaskAnnotation3DConfig(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
        idx, u, v = _backproject(pts, img_w=640, img_h=480, cfg=cfg)
        assert len(idx) == len(pts)
        assert np.all(u >= 0) and np.all(u < 640)
        assert np.all(v >= 0) and np.all(v < 480)

    def test_negative_z_filtered_out(self):
        pts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]], dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.ones((2, 3)))
        pts_arr, _ = _extract_cloud_arrays(pcd)
        idx, _, _ = _backproject(pts_arr, 640, 480, ObjectMaskAnnotation3DConfig())
        assert 0 not in idx

    def test_out_of_image_filtered_out(self):
        pts = np.array([[100.0, 0.0, 0.1]], dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.ones((1, 3)))
        pts_arr, _ = _extract_cloud_arrays(pcd)
        idx, _, _ = _backproject(pts_arr, 64, 48,
                                  ObjectMaskAnnotation3DConfig(fx=525.0, cx=32.0, cy=24.0))
        assert len(idx) == 0


# ── Front-view synthesis ──────────────────────────────────────────────────────


class TestFrontViewSynthesis:
    def test_canvas_size_derived_from_intrinsics(self, small_point_cloud):
        pts, cols = _extract_cloud_arrays(small_point_cloud)
        cfg = ObjectMaskAnnotation3DConfig(fx=100.0, fy=100.0, cx=32.0, cy=24.0)
        img = _render_front_view(pts, cols, cfg)
        assert img.width == int(cfg.cx * 2) + 1
        assert img.height == int(cfg.cy * 2) + 1

    def test_returns_pil_image(self, small_point_cloud):
        pts, cols = _extract_cloud_arrays(small_point_cloud)
        img = _render_front_view(pts, cols, ObjectMaskAnnotation3DConfig(
            fx=100.0, fy=100.0, cx=32.0, cy=24.0
        ))
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_empty_cloud_returns_black_canvas(self):
        pts = np.zeros((0, 3), dtype=np.float64)
        cols = np.zeros((0, 3), dtype=np.float64)
        cfg = ObjectMaskAnnotation3DConfig(fx=100.0, fy=100.0, cx=32.0, cy=24.0)
        img = _render_front_view(pts, cols, cfg)
        assert np.all(np.asarray(img) == 0)


# ── Handle — image_input=None uses front-view synthesis ──────────────────────


class TestNoImageInput:
    def test_handle_without_image_runs_successfully(self, small_point_cloud, full_mask_48x64):
        cfg = ObjectMaskAnnotation3DConfig(fx=100.0, fy=100.0, cx=32.0, cy=24.0)
        det = _make_detection(
            np.ones((int(cfg.cy * 2) + 1, int(cfg.cx * 2) + 1), dtype=bool)
        )
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[det]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=None,
                    model_backend="/fake/model.pt",
                    advanced_config=cfg,
                )
            )
        assert isinstance(result.annotations, list)


# ── Handle — zero detections ──────────────────────────────────────────────────


class TestNoDetections:
    def test_empty_annotations_when_no_detections(self, dummy_image_bytes, small_point_cloud):
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                )
            )
        assert result.annotations == []
        assert result.annotated_cloud is None

    def test_annotated_cloud_returned_with_no_detections_when_requested(
        self, dummy_image_bytes, small_point_cloud
    ):
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                    return_annotated_cloud=True,
                )
            )
        assert result.annotated_cloud is not None


# ── Handle — output structure ─────────────────────────────────────────────────


class TestAnnotationOutputStructure:
    def test_annotation_fields_present(self, dummy_image_bytes, small_point_cloud, full_mask_48x64):
        det = _make_detection(full_mask_48x64)
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[det]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                )
            )
        ann = result.annotations[0]
        assert ann.label == "chair"
        assert ann.class_id == 56
        assert ann.confidence == pytest.approx(0.85)
        assert len(ann.bbox_2d) == 4
        assert ann.mask_2d.dtype == bool

    def test_full_mask_attributes_inbounds_points(self, dummy_image_bytes, small_point_cloud):
        cfg = ObjectMaskAnnotation3DConfig(fx=100.0, fy=100.0, cx=32.0, cy=24.0)
        det = _make_detection(np.ones((49, 65), dtype=bool))
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[det]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                    advanced_config=cfg,
                )
            )
        assert len(result.annotations[0].point_indices) > 0

    def test_empty_mask_yields_zero_points(
        self, dummy_image_bytes, small_point_cloud, empty_mask_48x64
    ):
        det = _make_detection(empty_mask_48x64)
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[det]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                )
            )
        assert result.annotations[0].point_indices == []

    def test_multiple_detections_sorted_by_confidence(
        self, dummy_image_bytes, small_point_cloud, full_mask_48x64
    ):
        dets = [
            _make_detection(full_mask_48x64, label="person", class_id=0, conf=0.50),
            _make_detection(full_mask_48x64, label="chair", class_id=56, conf=0.90),
        ]
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=dets):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                )
            )
        confs = [a.confidence for a in result.annotations]
        assert confs == sorted(confs, reverse=True)

    def test_point_coords_shape(self, dummy_image_bytes, small_point_cloud, full_mask_48x64):
        det = _make_detection(full_mask_48x64)
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[det]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                )
            )
        ann = result.annotations[0]
        assert len(ann.point_coords) == len(ann.point_indices)
        assert all(len(c) == 3 for c in ann.point_coords)


# ── Optional outputs ──────────────────────────────────────────────────────────


class TestOptionalOutputs:
    def test_object_cloud_absent_by_default(
        self, dummy_image_bytes, small_point_cloud, full_mask_48x64
    ):
        det = _make_detection(full_mask_48x64)
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[det]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                )
            )
        assert result.annotations[0].object_cloud is None

    def test_object_cloud_returned_when_requested(
        self, dummy_image_bytes, small_point_cloud, full_mask_48x64
    ):
        cfg = ObjectMaskAnnotation3DConfig(fx=100.0, fy=100.0, cx=32.0, cy=24.0)
        det = _make_detection(np.ones((49, 65), dtype=bool))
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[det]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                    return_object_clouds=True,
                    advanced_config=cfg,
                )
            )
        ann = result.annotations[0]
        if ann.point_indices:
            assert isinstance(ann.object_cloud, o3d.geometry.PointCloud)

    def test_annotated_cloud_has_same_point_count(
        self, dummy_image_bytes, small_point_cloud, full_mask_48x64
    ):
        det = _make_detection(full_mask_48x64)
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[det]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                    return_annotated_cloud=True,
                )
            )
        n_in = len(np.asarray(small_point_cloud.points))
        n_out = len(np.asarray(result.annotated_cloud.points))
        assert n_out == n_in

    def test_object_cloud_preserves_original_colours(
        self, dummy_image_bytes, small_point_cloud
    ):
        """Extracted object cloud uses original colours, not annotated palette."""
        cfg = ObjectMaskAnnotation3DConfig(fx=100.0, fy=100.0, cx=32.0, cy=24.0)
        det = _make_detection(np.ones((49, 65), dtype=bool))
        with patch.object(ObjectMaskAnnotation3DHandler, "_run_yolo", return_value=[det]):
            result = ObjectMaskAnnotation3DHandler().handle(
                ObjectMaskAnnotation3DCommand(
                    point_cloud=small_point_cloud,
                    image_input=dummy_image_bytes,
                    model_backend="/fake/model.pt",
                    return_object_clouds=True,
                    return_annotated_cloud=True,
                    advanced_config=cfg,
                )
            )
        ann = result.annotations[0]
        if ann.point_indices and ann.object_cloud is not None:
            orig_cols = np.asarray(small_point_cloud.colors)[ann.point_indices]
            obj_cols = np.asarray(ann.object_cloud.colors)
            assert np.allclose(orig_cols, obj_cols), (
                "Object cloud colours must match original point cloud, not the annotated palette"
            )
