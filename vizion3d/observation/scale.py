"""Pure scale-estimation helpers for the ScaleObservation task."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .defaults import (
    CALIBRATED_SCALE_CORRECTION_BY_LABEL_DIM,
    DEFAULT_DIMENSION_RELIABILITY,
    DIMENSION_RELIABILITY_BY_LABEL,
    PRIOR_SOURCE_BY_LABEL,
    SCALE_SIZE_PRIORS_M,
    SIZE_PRIOR_ALIASES,
)
from .models import ObjectScaleObservation, ScaleCandidate, ScaleObservationConfig


def clean_object_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 40:
        return pts

    depth = -pts[:, 2]
    valid_depth = np.isfinite(depth) & (depth > 0)
    pts = pts[valid_depth]
    depth = depth[valid_depth]
    if pts.shape[0] < 40:
        return pts

    depth_med = float(np.median(depth))
    depth_mad = float(np.median(np.abs(depth - depth_med)))
    if depth_mad > 1e-9:
        depth_keep = np.abs(depth - depth_med) <= max(3.5 * depth_mad, 0.08 * depth_med)
        pts = pts[depth_keep]
    if pts.shape[0] < 40:
        return pts

    lo = np.percentile(pts, 2.0, axis=0)
    hi = np.percentile(pts, 98.0, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[keep]


def object_relative_dimensions(points: np.ndarray) -> dict[str, float]:
    """Return object dimensions in unscaled point-cloud units."""

    clean = clean_object_points(points)
    if clean.shape[0] < 40:
        return {"width": math.nan, "height": math.nan, "depth": math.nan}
    lo = np.percentile(clean, 10.0, axis=0)
    hi = np.percentile(clean, 90.0, axis=0)
    dims = np.maximum(hi - lo, 0.0)
    return {"width": float(dims[0]), "height": float(dims[1]), "depth": float(dims[2])}


def canonical_scale_label(label: str) -> str:
    normalized = " ".join(str(label).strip().lower().split())
    return SIZE_PRIOR_ALIASES.get(normalized, normalized)


def size_prior_for_label(label: str) -> tuple[str, dict[str, Any] | None, str]:
    canonical = canonical_scale_label(label)
    prior = SCALE_SIZE_PRIORS_M.get(canonical)
    return canonical, prior, PRIOR_SOURCE_BY_LABEL.get(canonical, "unknown")


def dimension_reliability(label: str, dimension: str) -> float:
    canonical = canonical_scale_label(label)
    return DIMENSION_RELIABILITY_BY_LABEL.get(canonical, DEFAULT_DIMENSION_RELIABILITY).get(
        dimension, DEFAULT_DIMENSION_RELIABILITY[dimension]
    )


def calibration_factor(label: str, dimension: str) -> float:
    canonical = canonical_scale_label(label)
    return CALIBRATED_SCALE_CORRECTION_BY_LABEL_DIM.get(canonical, {}).get(dimension, 1.0)


def prior_uncertainty_score(mean: float, sigma: float) -> float:
    if mean <= 0 or not np.isfinite(sigma):
        return 0.25
    coefficient = max(0.0, sigma / mean)
    if coefficient >= 0.75:
        return 0.20
    return float(max(0.25, min(1.0, 1.0 / (1.0 + 2.5 * coefficient))))


def _fallback_image_size_from_bbox(bbox: list[float]) -> tuple[int, int]:
    if len(bbox) != 4:
        return (1, 1)
    return (
        max(1, int(math.ceil(max(bbox[0], bbox[2])))),
        max(1, int(math.ceil(max(bbox[1], bbox[3])))),
    )


def _resolve_image_size(
    image_size: tuple[int, int] | None,
    bbox: list[float],
    mask: Any,
) -> tuple[int, int]:
    if image_size is not None:
        return image_size
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.ndim >= 2 and mask_array.shape[0] > 1 and mask_array.shape[1] > 1:
            return int(mask_array.shape[1]), int(mask_array.shape[0])
    return _fallback_image_size_from_bbox(bbox)


def bbox_features(bbox: list[float], image_size: tuple[int, int]) -> dict[str, Any]:
    if len(bbox) != 4:
        return {
            "area_ratio": 0.0,
            "touches_edge": True,
            "horizontal_band": "unknown",
            "vertical_band": "unknown",
        }
    x1, y1, x2, y2 = bbox
    width, height = image_size
    margin = 0.02 * max(width, height)
    touches_edge = x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)
    area_ratio = bbox_width * bbox_height / max(float(width * height), 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    horizontal_band = (
        "left" if cx < width / 3.0 else "right" if cx > 2.0 * width / 3.0 else "center"
    )
    vertical_band = (
        "top" if cy < height / 3.0 else "bottom" if cy > 2.0 * height / 3.0 else "middle"
    )
    return {
        "area_ratio": float(area_ratio),
        "touches_edge": bool(touches_edge),
        "horizontal_band": horizontal_band,
        "vertical_band": vertical_band,
    }


def mask_features(mask: Any, bbox: list[float], image_size: tuple[int, int]) -> dict[str, float]:
    if mask is None:
        return {"mask_area_ratio": 0.0, "mask_bbox_fill": 0.0}
    mask_array = np.asarray(mask, dtype=bool)
    image_width, image_height = image_size
    image_area = max(float(image_width * image_height), 1.0)
    mask_area = float(mask_array.sum()) if mask_array.size else 0.0
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        bbox_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    else:
        bbox_area = 0.0
    return {
        "mask_area_ratio": mask_area / image_area,
        "mask_bbox_fill": mask_area / max(bbox_area, 1.0),
    }


def bbox_area_ratio(bbox: list[float], image_size: tuple[int, int] | None) -> float | None:
    if image_size is None:
        return None
    return float(bbox_features(bbox, image_size)["area_ratio"])


def bbox_touches_edge(bbox: list[float], image_size: tuple[int, int] | None) -> bool:
    if image_size is None or len(bbox) != 4:
        return True
    return bool(bbox_features(bbox, image_size)["touches_edge"])


def position_quality(points: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    centroid = points.mean(axis=0)
    relative_depth = max(float(-centroid[2]), 1e-6)
    off_axis_x = math.atan2(abs(float(centroid[0])), relative_depth)
    off_axis_y = math.atan2(abs(float(centroid[1])), relative_depth)
    off_axis_quality = 1.0 / (1.0 + off_axis_x + 0.5 * off_axis_y)
    distance_quality = 1.0 / (1.0 + max(relative_depth - 4.0, 0.0) * 0.15)
    return max(0.1, min(1.0, off_axis_quality * distance_quality))


def depth_band(points: np.ndarray) -> str:
    if points.shape[0] == 0:
        return "unknown"
    depth = max(float(-np.mean(points[:, 2])), 0.0)
    if depth < 1.5:
        return "near"
    if depth <= 4.0:
        return "mid"
    return "far"


def depth_spread_quality(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] < 40:
        return 0.0, math.inf
    depth = -points[:, 2]
    median_depth = max(float(np.median(depth)), 1e-6)
    spread = float(np.percentile(depth, 95.0) - np.percentile(depth, 5.0))
    ratio = spread / median_depth
    if ratio <= 0.15:
        return 1.0, ratio
    if ratio <= 0.35:
        return 0.7, ratio
    if ratio <= 0.65:
        return 0.35, ratio
    return 0.05, ratio


def scene_scale_plausibility(scale: float, scene_bounds: dict[str, Any] | None) -> float:
    if scene_bounds is None or not np.isfinite(scale) or scale <= 0:
        return 1.0

    width = float(scene_bounds.get("width_m", math.nan)) * scale
    height = float(scene_bounds.get("height_m", math.nan)) * scale
    length = float(scene_bounds.get("length_m", math.nan)) * scale
    dims = [width, height, length]
    if not all(np.isfinite(value) and value > 0 for value in dims):
        return 1.0

    score = 1.0
    if height > 5.0:
        score *= max(0.10, 5.0 / height)
    if width > 14.0:
        score *= max(0.10, 14.0 / width)
    if length > 14.0:
        score *= max(0.10, 14.0 / length)
    if height < 1.2:
        score *= max(0.20, height / 1.2)
    if width < 1.0:
        score *= max(0.20, width / 1.0)
    if length < 1.0:
        score *= max(0.20, length / 1.0)
    volume = width * height * length
    if volume > 600.0:
        score *= max(0.05, 600.0 / volume)
    return max(0.02, min(1.0, score))


def object_axis_agreement(
    label: str,
    dims: dict[str, float],
    prior: dict[str, Any] | None,
) -> tuple[float, str | None]:
    if prior is None:
        return 0.0, "missing_size_prior"

    logs: list[float] = []
    weights: list[float] = []
    for dim_name, observed in dims.items():
        if not np.isfinite(observed) or observed <= 1e-6:
            continue
        dim_weight = dimension_reliability(label, dim_name)
        if dim_weight < 0.25:
            continue
        prior_mean, prior_sigma = prior[dim_name]
        corrected_scale = float(prior_mean) / observed * calibration_factor(label, dim_name)
        if np.isfinite(corrected_scale) and corrected_scale > 0:
            logs.append(math.log(corrected_scale))
            weights.append(dim_weight * prior_uncertainty_score(prior_mean, prior_sigma))

    if len(logs) < 2:
        return 0.65, None

    values = np.asarray(logs, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    center = np.average(values, weights=weight_values)
    spread = float(np.sqrt(np.average((values - center) ** 2, weights=weight_values)))
    if spread <= 0.20:
        return 1.0, None
    if spread <= 0.40:
        return 0.75, None
    if spread <= 0.65:
        return 0.40, "weak_multi_axis_agreement"
    return 0.10, "object_dimensions_disagree"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = np.maximum(weights[order], 0.0)
    total = float(ordered_weights.sum())
    if total <= 0:
        return float(np.quantile(ordered_values, q))
    target = total * q
    cumulative = np.cumsum(ordered_weights)
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[max(0, min(index, len(ordered_values) - 1))])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return weighted_quantile(values, weights, 0.5)


def huber_location(values: np.ndarray, weights: np.ndarray, delta: float) -> float:
    center = weighted_median(values, weights)
    for _ in range(8):
        residual = values - center
        factors = np.where(
            np.abs(residual) <= delta,
            1.0,
            delta / np.maximum(np.abs(residual), 1e-12),
        )
        adjusted = np.maximum(weights, 0.0) * factors
        total = float(adjusted.sum())
        if total <= 0:
            return float(center)
        next_center = float(np.average(values, weights=adjusted))
        if abs(next_center - center) < 1e-6:
            break
        center = next_center
    return float(center)


def family_balanced_location(
    candidates: list[ScaleCandidate],
    logs: np.ndarray,
    weights: np.ndarray,
    *,
    family: str,
    huber_delta: float,
) -> float:
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for candidate, log_value, weight in zip(candidates, logs, weights):
        if family == "dimension":
            key = candidate.dimension
        elif family == "label":
            key = candidate.canonical_label or canonical_scale_label(candidate.label)
        elif family == "source":
            key = candidate.prior_source
        else:
            key = "all"
        group_values, group_weights = grouped.setdefault(key, ([], []))
        group_values.append(float(log_value))
        group_weights.append(float(weight))

    centers: list[float] = []
    center_weights: list[float] = []
    for group_values, group_weights in grouped.values():
        value_array = np.asarray(group_values, dtype=np.float64)
        weight_array = np.asarray(group_weights, dtype=np.float64)
        centers.append(huber_location(value_array, weight_array, huber_delta))
        center_weights.append(math.sqrt(max(float(weight_array.sum()), 1e-12)))
    if not centers:
        return float(np.average(logs, weights=weights))
    return float(
        np.average(
            np.asarray(centers, dtype=np.float64),
            weights=np.asarray(center_weights, dtype=np.float64),
        )
    )


def aggregate_log_scale(
    candidates: list[ScaleCandidate],
    logs: np.ndarray,
    weights: np.ndarray,
    config: ScaleObservationConfig,
) -> float:
    if config.aggregate == "lower_quantile_mean_blend":
        lower_log = weighted_quantile(logs, weights, 0.35)
        mean_log = float(np.average(logs, weights=weights))
        return 0.60 * mean_log + 0.40 * lower_log
    if config.aggregate == "median_mean_blend":
        return float(
            0.65 * weighted_median(logs, weights) + 0.35 * np.average(logs, weights=weights)
        )
    if config.aggregate == "dimension_class_balanced_huber":
        dim_log = family_balanced_location(
            candidates, logs, weights, family="dimension", huber_delta=config.huber_delta
        )
        label_log = family_balanced_location(
            candidates, logs, weights, family="label", huber_delta=config.huber_delta
        )
        return 0.5 * dim_log + 0.5 * label_log
    if config.aggregate == "dimension_class_trimmed_huber":
        if len(logs) > 3:
            lo = weighted_quantile(logs, weights, config.winsor_quantile)
            hi = weighted_quantile(logs, weights, 1.0 - config.winsor_quantile)
            keep = (logs >= lo) & (logs <= hi)
            if int(np.count_nonzero(keep)) >= 2:
                kept_candidates = [c for c, keep_item in zip(candidates, keep) if bool(keep_item)]
                return aggregate_log_scale(
                    kept_candidates,
                    logs[keep],
                    weights[keep],
                    config.model_copy(update={"aggregate": "dimension_class_balanced_huber"}),
                )
        return aggregate_log_scale(
            candidates,
            logs,
            weights,
            config.model_copy(update={"aggregate": "dimension_class_balanced_huber"}),
        )
    if config.aggregate == "mean":
        return float(np.average(logs, weights=weights))
    return weighted_median(logs, weights)


def candidate_weight(candidate: ScaleCandidate, config: ScaleObservationConfig) -> float:
    base = max(float(candidate.weight), 0.0)
    point_factor = min(1.0, math.log1p(max(candidate.clean_point_count, 0)) / math.log1p(5000.0))
    edge_factor = 0.45 if candidate.touches_edge else 1.0
    quality_term = (
        max(candidate.object_quality, 1e-6)
        * max(candidate.dimension_reliability, 1e-6)
        * max(candidate.prior_uncertainty_score, 1e-6)
        * max(candidate.axis_agreement_score, 1e-6)
        * max(candidate.scene_plausibility_score, 1e-6)
        * max(point_factor, 1e-6)
        * edge_factor
    )
    return base * (quality_term**config.quality_power)


def build_candidates_from_annotations(
    annotations: list[Any],
    *,
    image_size: tuple[int, int] | None,
    scene_bounds: dict[str, Any] | None = None,
) -> tuple[list[ScaleCandidate], list[ObjectScaleObservation]]:
    candidates: list[ScaleCandidate] = []
    observations: list[ObjectScaleObservation] = []

    for instance_id, ann in enumerate(annotations):
        label = str(getattr(ann, "label", ""))
        canonical_label, prior, prior_source = size_prior_for_label(label)
        points = np.asarray(getattr(ann, "point_coords", []), dtype=np.float64)
        clean_points = clean_object_points(points)
        dims = object_relative_dimensions(clean_points)
        axis_score, axis_rejection = object_axis_agreement(canonical_label, dims, prior)
        bbox = [float(v) for v in getattr(ann, "bbox_2d", [])]
        mask = getattr(ann, "mask_2d", None)
        ann_image_size = _resolve_image_size(image_size, bbox, mask)
        bbox_info = bbox_features(bbox, ann_image_size)
        mask_info = mask_features(mask, bbox, ann_image_size)
        point_count = int(points.shape[0])
        clean_point_count = int(clean_points.shape[0])
        area_ratio = float(bbox_info["area_ratio"])
        touches_edge = bool(bbox_info["touches_edge"])
        rejection_reasons: list[str] = []
        if prior is None:
            rejection_reasons.append("missing_size_prior")
        if point_count < 160:
            rejection_reasons.append("too_few_raw_points")
        if clean_point_count < 120:
            rejection_reasons.append("too_few_clean_points")
        if area_ratio < 0.005:
            rejection_reasons.append("bbox_too_small")
        if mask_info["mask_area_ratio"] < 0.0025:
            rejection_reasons.append("mask_too_small")
        if mask_info["mask_bbox_fill"] < 0.12:
            rejection_reasons.append("weak_mask_bbox_fill")
        if touches_edge:
            rejection_reasons.append("bbox_touches_image_edge")
        finite_dims = [value for value in dims.values() if np.isfinite(value) and value > 1e-6]
        if len(finite_dims) < 2:
            rejection_reasons.append("degenerate_object_dimensions")
        spread_score, spread_ratio = depth_spread_quality(clean_points)
        if spread_ratio > 0.50:
            rejection_reasons.append("excessive_internal_depth_spread")
        if axis_rejection in {"object_dimensions_disagree", "weak_multi_axis_agreement"}:
            rejection_reasons.append(axis_rejection)

        confidence = float(getattr(ann, "confidence", 0.0))
        point_score = min(1.0, math.log10(max(clean_point_count, 1)) / 4.0)
        area_score = min(1.0, max(0.0, area_ratio / 0.05))
        mask_fill_score = min(1.0, max(0.15, mask_info["mask_bbox_fill"]))
        pos_score = position_quality(clean_points)
        edge_score = 0.15 if touches_edge else 1.0
        object_quality = (
            confidence
            * point_score
            * max(0.15, area_score)
            * mask_fill_score
            * pos_score
            * edge_score
            * spread_score
        )
        object_quality = max(0.0, min(1.0, object_quality))
        if axis_rejection:
            object_quality *= axis_score
        centroid = (
            np.mean(clean_points, axis=0).tolist()
            if clean_points.shape[0] > 0
            else [math.nan, math.nan, math.nan]
        )

        observations.append(
            ObjectScaleObservation(
                instance_id=instance_id,
                label=label,
                canonical_label=canonical_label,
                prior_source=prior_source if prior is not None else "missing",
                class_id=getattr(ann, "class_id", None),
                confidence=confidence,
                bbox_2d=bbox,
                bbox_area_ratio=area_ratio,
                touches_edge=touches_edge,
                point_count=point_count,
                clean_point_count=clean_point_count,
                centroid=centroid,
                depth_band=depth_band(clean_points),
                horizontal_band=str(bbox_info["horizontal_band"]),
                vertical_band=str(bbox_info["vertical_band"]),
                observed_dimensions=dims,
                mask_area_ratio=float(mask_info["mask_area_ratio"]),
                mask_bbox_fill=float(mask_info["mask_bbox_fill"]),
                depth_spread_ratio=float(spread_ratio),
                axis_agreement_score=axis_score,
                prior_available=prior is not None,
                accepted=not rejection_reasons,
                rejection_reasons=rejection_reasons,
                quality=object_quality,
            )
        )

        if prior is None or rejection_reasons:
            continue

        base_weight = float(prior["r"]) * object_quality
        for dimension, observed in dims.items():
            if not np.isfinite(observed) or observed <= 1e-6:
                continue
            prior_mean, prior_sigma = prior[dimension]
            dim_weight = dimension_reliability(canonical_label, dimension)
            if dim_weight < 0.10:
                continue
            uncertainty = prior_uncertainty_score(float(prior_mean), float(prior_sigma))
            if uncertainty < 0.25 and dim_weight < 0.40:
                continue
            correction = calibration_factor(canonical_label, dimension)
            scale = float(prior_mean) / observed * correction
            if not np.isfinite(scale) or scale <= 0:
                continue
            plausibility = scene_scale_plausibility(scale, scene_bounds)
            if plausibility < 0.08:
                continue
            candidates.append(
                ScaleCandidate(
                    label=label,
                    canonical_label=canonical_label,
                    prior_source=prior_source,
                    dimension=dimension,
                    observed_relative=float(observed),
                    prior_m=float(prior_mean) * correction,
                    scale=scale,
                    weight=max(
                        1e-6,
                        base_weight * dim_weight * uncertainty * axis_score * plausibility,
                    ),
                    instance_id=instance_id,
                    class_id=getattr(ann, "class_id", None),
                    object_quality=object_quality,
                    dimension_reliability=dim_weight,
                    point_count=point_count,
                    clean_point_count=clean_point_count,
                    bbox_area_ratio=area_ratio,
                    touches_edge=touches_edge,
                    calibration_factor=correction,
                    prior_sigma_m=float(prior_sigma),
                    prior_uncertainty_score=uncertainty,
                    axis_agreement_score=axis_score,
                    scene_plausibility_score=plausibility,
                )
            )

    return candidates, observations


def estimate_scale(
    candidates: list[ScaleCandidate],
    config: ScaleObservationConfig,
    scene_bounds: dict[str, Any] | None = None,
) -> tuple[float, float, str, list[ScaleCandidate]]:
    if config.candidate_source == "all":
        selected = [c for c in candidates if np.isfinite(c.scale) and c.scale > 0]
    elif config.candidate_source == "usable":
        selected = [
            c
            for c in candidates
            if not c.touches_edge
            and c.clean_point_count >= 120
            and np.isfinite(c.scale)
            and c.scale > 0
        ]
    elif config.candidate_source == "yoloe_strong":
        selected = [
            c
            for c in candidates
            if not c.touches_edge
            and c.clean_point_count >= 400
            and c.axis_agreement_score >= 0.25
            and c.prior_source in {"coco", "yoloe_pf"}
            and np.isfinite(c.scale)
            and c.scale > 0
        ]
    else:
        selected = [
            c
            for c in candidates
            if not c.touches_edge
            and c.clean_point_count >= 500
            and c.axis_agreement_score >= 0.2
            and np.isfinite(c.scale)
            and c.scale > 0
        ]

    logs: list[float] = []
    weights: list[float] = []
    accepted: list[ScaleCandidate] = []
    for candidate in selected:
        weight = candidate_weight(candidate, config)
        if weight < config.min_candidate_weight:
            candidate.rejection_reason = "below_variant_weight_threshold"
            continue
        logs.append(math.log(candidate.scale))
        weights.append(weight)
        accepted.append(candidate)

    for candidate in candidates:
        candidate.accepted = candidate in accepted
        if not candidate.accepted and candidate.rejection_reason is None:
            candidate.rejection_reason = "not_selected_by_variant"

    if not logs:
        reason = (
            f"scale_variant={config.name}; no usable candidates; "
            f"fallback={config.no_candidate}; scale={config.prior:.4f}."
        )
        return float(config.prior), 0.05, reason, candidates

    log_array = np.asarray(logs, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    raw_log = aggregate_log_scale(accepted, log_array, weight_array, config)
    raw_scale = math.exp(raw_log)

    spread = float(np.std(log_array)) if len(log_array) > 1 else 0.0
    confidence = min(
        1.0,
        (len(logs) / 8.0)
        * (1.0 / (1.0 + 4.0 * spread))
        * (1.0 + 0.10 * max(len({c.canonical_label or c.label for c in accepted}) - 1, 0))
        * (1.0 + 0.05 * max(len({c.dimension for c in accepted}) - 1, 0)),
    )
    confidence = min(1.0, max(0.0, confidence**config.confidence_power))
    object_weight = max(float(sum(weights)) * max(confidence, 0.02), config.min_object_weight)
    object_weight = object_weight**config.object_weight_power
    prior_weight = config.prior_weight * ((1.0 - confidence) ** config.prior_weight_power)
    scale_factor = math.exp(
        (raw_log * object_weight + math.log(config.prior) * prior_weight)
        / (object_weight + prior_weight)
    )
    reason = (
        f"scale_variant={config.name}; selected={len(accepted)}; raw_scale={raw_scale:.4f}; "
        f"object_weight={object_weight:.6f}; prior={config.prior:.4f}; "
        f"prior_weight={prior_weight:.6f}; aggregate={config.aggregate}; "
        f"source={config.candidate_source}."
    )
    return float(scale_factor), float(confidence), reason, candidates
