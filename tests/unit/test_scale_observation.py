from __future__ import annotations

import json

import numpy as np
import pytest

from vizion3d.annotation.models import MaskAnnotation3D
from vizion3d.lifting.utils import create_ply_binary
from vizion3d.observation import (
    ScaleObservation,
    ScaleObservationAdvancedConfig,
    ScaleObservationCommand,
    ScaleObservationConfig,
)
from vizion3d.observation.defaults import (
    CALIBRATED_SCALE_CORRECTION_BY_LABEL_DIM,
    COCO_SIZE_PRIORS_M,
    DIMENSION_RELIABILITY_BY_LABEL,
)
from vizion3d.observation.scale import (
    build_candidates_from_annotations,
    estimate_scale,
    prior_uncertainty_score,
)

o3d = pytest.importorskip("open3d", reason="open3d required")


def _chair_points(count: int = 1200) -> np.ndarray:
    rng = np.random.default_rng(123)
    width = 0.50 * 0.6804 / 0.5
    height = 0.85 * 0.5693 / 0.5
    depth = 0.55 * 0.5350 / 0.5
    pts = rng.random((count, 3), dtype=np.float64)
    pts[:, 0] = (pts[:, 0] - 0.5) * width
    pts[:, 1] = (pts[:, 1] - 0.5) * height
    pts[:, 2] = -2.0 + (pts[:, 2] - 0.5) * depth
    return pts


def _point_cloud(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.full((len(points), 3), 0.5))
    return pcd


def _annotation(points: np.ndarray, *, bbox=None, label="chair") -> MaskAnnotation3D:
    return MaskAnnotation3D(
        label=label,
        class_id=56,
        confidence=0.9,
        bbox_2d=bbox or [20.0, 20.0, 80.0, 80.0],
        mask_2d=np.ones((100, 100), dtype=bool),
        point_indices=list(range(len(points))),
        point_coords=points.tolist(),
    )


def test_default_config_is_promoted_v4_402():
    cfg = ScaleObservationConfig()
    assert cfg.name == "v4_iter_402_lower_quantile_mean_blend"
    assert cfg.candidate_source == "strong"
    assert cfg.aggregate == "lower_quantile_mean_blend"
    assert cfg.prior == pytest.approx(0.30)


def test_build_candidates_accepts_annotation_task_annotations():
    ann = _annotation(_chair_points())
    candidates, observations = build_candidates_from_annotations([ann], image_size=(100, 100))
    assert observations[0].label == "chair"
    assert observations[0].accepted is True
    assert {c.dimension for c in candidates} == {"width", "height", "depth"}


def test_v4_tables_cover_all_research_priors():
    assert set(DIMENSION_RELIABILITY_BY_LABEL) == set(COCO_SIZE_PRIORS_M)
    assert set(CALIBRATED_SCALE_CORRECTION_BY_LABEL_DIM) == set(COCO_SIZE_PRIORS_M) - {"mouse"}


def test_v4_prior_uncertainty_matches_promoted_formula():
    assert prior_uncertainty_score(1.0, 0.0) == pytest.approx(1.0)
    assert prior_uncertainty_score(1.0, 0.2) == pytest.approx(1.0 / 1.5)
    assert prior_uncertainty_score(1.0, 0.8) == pytest.approx(0.20)


def test_build_candidates_rejects_weak_mask_fill_like_v4():
    ann = _annotation(_chair_points())
    ann.mask_2d[:] = False
    candidates, observations = build_candidates_from_annotations([ann], image_size=(100, 100))
    assert candidates == []
    assert "mask_too_small" in observations[0].rejection_reasons
    assert "weak_mask_bbox_fill" in observations[0].rejection_reasons


def test_estimate_scale_filters_edge_touching_candidates():
    ann = _annotation(_chair_points(), bbox=[0.0, 20.0, 80.0, 80.0])
    candidates, _ = build_candidates_from_annotations([ann], image_size=(100, 100))
    scale, confidence, reason, candidates = estimate_scale(candidates, ScaleObservationConfig())
    assert scale == pytest.approx(0.30)
    assert confidence == pytest.approx(0.05)
    assert "no usable candidates" in reason
    assert all(not c.accepted for c in candidates)


def test_estimate_scale_applies_scene_extent_guard_when_enabled():
    ann = _annotation(_chair_points())
    candidates, _ = build_candidates_from_annotations(
        [ann],
        image_size=(100, 100),
        scene_bounds={"width_m": 100.0, "height_m": 2.0, "length_m": 2.0},
    )
    cfg = ScaleObservationConfig(scene_extent_guard_strength=1.0, max_scene_extent_m=5.8)
    scale, _, _, _ = estimate_scale(
        candidates,
        cfg,
        {"width_m": 100.0, "height_m": 2.0, "length_m": 2.0},
    )
    assert scale == pytest.approx(0.058)


def test_direct_scale_observation_returns_scaled_point_cloud():
    points = _chair_points()
    result = ScaleObservation().run(
        ScaleObservationCommand(
            point_cloud=_point_cloud(points),
            annotations=[_annotation(points)],
            return_scaled_point_cloud=True,
            advanced_config=ScaleObservationAdvancedConfig(image_width=100, image_height=100),
        )
    )
    assert result.algorithm_version == "v4_iter_402_lower_quantile_mean_blend"
    assert result.accepted_candidates > 0
    assert result.scale_factor == pytest.approx(0.5, rel=0.15)
    assert result.scaled_point_cloud is not None
    original = np.asarray(_point_cloud(points).points)
    scaled = np.asarray(result.scaled_point_cloud.points)
    assert np.max(np.abs(scaled - original * result.scale_factor)) < 1e-9


def test_direct_scale_observation_accepts_serialized_annotation_dict():
    points = _chair_points()
    ann_dict = _annotation(points).model_dump()
    ann_dict["mask_2d"] = ann_dict["mask_2d"].tolist()
    result = ScaleObservation().run(
        ScaleObservationCommand(
            point_cloud=_point_cloud(points),
            annotations=[ann_dict],
            advanced_config=ScaleObservationAdvancedConfig(image_width=100, image_height=100),
        )
    )
    assert result.accepted_candidates > 0


def test_rest_scale_observation_endpoint():
    from fastapi.testclient import TestClient

    from vizion3d.server.rest.app import app

    points = _chair_points()
    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    ply = create_ply_binary(points.astype(np.float32), colors)
    ann_dict = _annotation(points).model_dump()
    ann_dict["mask_2d"] = ann_dict["mask_2d"].tolist()

    response = TestClient(app).post(
        "/observation/scale-observation",
        files={"point_cloud": ("scene.ply", ply, "application/octet-stream")},
        data={
            "annotations_json": json.dumps([ann_dict]),
            "image_width": "100",
            "image_height": "100",
            "return_scaled_point_cloud": "true",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["algorithm_version"] == "v4_iter_402_lower_quantile_mean_blend"
    assert payload["accepted_candidates"] > 0
    assert payload["scaled_point_cloud_ply"] is not None


def test_grpc_scale_observation_request():
    from vizion3d.proto import lifting_pb2
    from vizion3d.server.grpc.server import LiftingServiceServicer, _mask_to_png_bytes

    points = _chair_points()
    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    ply = create_ply_binary(points.astype(np.float32), colors)
    ann = _annotation(points)
    request = lifting_pb2.ScaleObservationRequest(
        point_cloud_ply=ply,
        return_scaled_point_cloud=True,
        return_report=True,
        image_width=100,
        image_height=100,
    )
    item = request.annotations.add(
        label=ann.label,
        class_id=ann.class_id,
        confidence=ann.confidence,
        bbox_2d=ann.bbox_2d,
        mask_image=_mask_to_png_bytes(ann.mask_2d),
    )
    for coord in ann.point_coords:
        item.point_coords.append(lifting_pb2.FloatRow(values=coord))

    response = LiftingServiceServicer().RunScaleObservation(request, None)
    assert response.algorithm_version == "v4_iter_402_lower_quantile_mean_blend"
    assert response.accepted_candidates > 0
    assert response.scaled_point_cloud_ply
