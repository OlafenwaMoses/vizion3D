"""Data models for the ScaleObservation task."""

from __future__ import annotations

from typing import Any

import numpy as np
from open3d.geometry import Image as O3dImage  # type: ignore[import-untyped]
from open3d.geometry import PointCloud as O3dPointCloud  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field


class ScaleObservationConfig(BaseModel):
    """Runtime configuration for the promoted ScaleObservation estimator."""

    name: str = "v4_1_yoloe_strong_dimension_class_trimmed_huber"
    candidate_source: str = "yoloe_strong"
    aggregate: str = "dimension_class_trimmed_huber"
    prior: float = 0.30
    prior_weight: float = 0.015
    quality_power: float = 0.20
    confidence_power: float = 1.0
    min_candidate_weight: float = 1e-6
    no_candidate: str = "prior"
    prior_weight_power: float = 1.75
    winsor_quantile: float = 0.08
    huber_delta: float = 0.12
    object_weight_power: float = 1.1
    min_object_weight: float = 0.02


class ScaleObservationAdvancedConfig(BaseModel):
    """Camera/image settings for ScaleObservation geometry outputs."""

    image_width: int | None = None
    image_height: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None


class ScaleCandidate(BaseModel):
    """Single object/dimension-derived scale candidate."""

    label: str
    canonical_label: str | None = None
    prior_source: str = "unknown"
    dimension: str
    observed_relative: float
    prior_m: float
    scale: float
    weight: float
    accepted: bool = False
    rejection_reason: str | None = None
    instance_id: int | None = None
    class_id: int | None = None
    object_quality: float = 1.0
    dimension_reliability: float = 1.0
    point_count: int = 0
    clean_point_count: int = 0
    bbox_area_ratio: float | None = None
    touches_edge: bool = False
    calibration_factor: float = 1.0
    prior_sigma_m: float | None = None
    prior_uncertainty_score: float = 1.0
    axis_agreement_score: float = 1.0
    scene_plausibility_score: float = 1.0


class ObjectScaleObservation(BaseModel):
    """Normalised per-object evidence used by ScaleObservation."""

    instance_id: int
    label: str
    canonical_label: str | None = None
    prior_source: str = "unknown"
    class_id: int | None
    confidence: float
    bbox_2d: list[float]
    bbox_area_ratio: float = 0.0
    touches_edge: bool = False
    point_count: int
    clean_point_count: int
    centroid: list[float] = Field(default_factory=list)
    depth_band: str = "unknown"
    horizontal_band: str = "unknown"
    vertical_band: str = "unknown"
    observed_dimensions: dict[str, float]
    mask_area_ratio: float = 0.0
    mask_bbox_fill: float = 0.0
    depth_spread_ratio: float = 0.0
    axis_agreement_score: float
    prior_available: bool
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    quality: float = 0.0


class ScaledDepthMetadata(BaseModel):
    """Metadata for a scaled depth image reprojected from a scaled point cloud."""

    dtype: str = "float32"
    units: str = "metres"
    invalid_value: float = 0.0
    coordinate_space: str = "OpenGL camera space: X right, Y up, Z negative forward"
    z_buffer_policy: str = "nearest point wins"


class ScaleObservationResult(BaseModel):
    """Result payload returned by ScaleObservation."""

    scale_factor: float
    scale_confidence: float
    scale_confidence_reason: str
    algorithm_version: str
    candidates: list[ScaleCandidate]
    object_observations: list[ObjectScaleObservation] = Field(default_factory=list)
    accepted_candidates: int
    rejected_candidates: int
    scaled_point_cloud: O3dPointCloud | None = None
    scaled_depth_image: O3dImage | None = None
    scaled_depth_metadata: ScaledDepthMetadata | None = None
    scale_report: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


def point_cloud_bounds(points: np.ndarray) -> dict[str, float]:
    """Return axis-aligned bounds for a point array."""

    if points.size == 0:
        return {"width_m": 0.0, "height_m": 0.0, "length_m": 0.0, "volume_m3": 0.0}
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    dims = np.maximum(maxs - mins, 0.0)
    return {
        "width_m": float(dims[0]),
        "height_m": float(dims[1]),
        "length_m": float(dims[2]),
        "volume_m3": float(dims[0] * dims[1] * dims[2]),
    }
