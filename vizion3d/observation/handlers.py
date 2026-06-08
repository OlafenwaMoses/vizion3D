"""CQRS command handler for the ScaleObservation task."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from vizion3d.annotation.models import MaskAnnotation3D
from vizion3d.core.cqrs import CommandHandler

from .commands import ScaleObservationCommand
from .models import ScaledDepthMetadata, ScaleObservationResult, point_cloud_bounds
from .scale import build_candidates_from_annotations, estimate_scale


class ScaleObservationHandler(CommandHandler[ScaleObservationCommand, ScaleObservationResult]):
    """Handle :class:`~vizion3d.observation.commands.ScaleObservationCommand`."""

    def handle(self, command: ScaleObservationCommand) -> ScaleObservationResult:
        point_cloud = _coerce_point_cloud(command.point_cloud)
        annotations = _coerce_annotations(command.annotations)
        image_size = _image_size(command)
        scene_bounds = point_cloud_bounds(np.asarray(point_cloud.points, dtype=np.float64))
        candidates, object_observations = build_candidates_from_annotations(
            annotations,
            image_size=image_size,
            scene_bounds=scene_bounds,
        )
        scale_factor, confidence, reason, candidates = estimate_scale(
            candidates,
            command.config,
            scene_bounds,
        )

        scaled_cloud = (
            _scale_point_cloud(point_cloud, scale_factor)
            if command.return_scaled_point_cloud
            else None
        )
        scaled_depth = None
        metadata = None
        if command.return_scaled_depth:
            source_cloud = (
                scaled_cloud
                if scaled_cloud is not None
                else _scale_point_cloud(point_cloud, scale_factor)
            )
            scaled_depth = _project_depth_image(source_cloud, command)
            metadata = ScaledDepthMetadata()

        accepted = sum(1 for c in candidates if c.accepted)
        rejected = len(candidates) - accepted
        report: dict[str, Any] = {}
        if command.return_report:
            pts = np.asarray(point_cloud.points, dtype=np.float64)
            report = {
                "algorithm_version": command.config.name,
                "scale_factor": scale_factor,
                "scale_confidence": confidence,
                "generated_bounds": point_cloud_bounds(pts),
                "scaled_bounds": point_cloud_bounds(pts * scale_factor),
                "accepted_candidates": [c.model_dump() for c in candidates if c.accepted],
                "rejected_candidates": [c.model_dump() for c in candidates if not c.accepted],
            }

        return ScaleObservationResult(
            scale_factor=scale_factor,
            scale_confidence=confidence,
            scale_confidence_reason=reason,
            algorithm_version=command.config.name,
            candidates=candidates,
            object_observations=object_observations,
            accepted_candidates=accepted,
            rejected_candidates=rejected,
            scaled_point_cloud=scaled_cloud,
            scaled_depth_image=scaled_depth,
            scaled_depth_metadata=metadata,
            scale_report=report,
        )


def _coerce_point_cloud(value):
    import open3d as o3d

    if isinstance(value, o3d.geometry.PointCloud):
        return value
    if isinstance(value, (bytes, bytearray)):
        fd, path = tempfile.mkstemp(suffix=".ply")
        try:
            os.write(fd, value)
            os.close(fd)
            return o3d.io.read_point_cloud(path)
        finally:
            os.unlink(path)
    if isinstance(value, str):
        return o3d.io.read_point_cloud(value)
    raise TypeError("point_cloud must be an Open3D PointCloud, PLY bytes, or a file path.")


def _coerce_annotations(values: list[MaskAnnotation3D] | list[dict[str, Any]] | None) -> list[Any]:
    if values is None:
        return []
    annotations: list[Any] = []
    for item in values:
        if isinstance(item, MaskAnnotation3D):
            annotations.append(item)
        elif isinstance(item, dict):
            data = dict(item)
            if "mask_2d" in data and not isinstance(data["mask_2d"], np.ndarray):
                data["mask_2d"] = np.asarray(data["mask_2d"], dtype=bool)
            annotations.append(MaskAnnotation3D(**data))
        else:
            annotations.append(item)
    return annotations


def _image_size(command: ScaleObservationCommand) -> tuple[int, int] | None:
    cfg = command.advanced_config
    if cfg.image_width is not None and cfg.image_height is not None:
        return int(cfg.image_width), int(cfg.image_height)
    for ann in command.annotations or []:
        mask = getattr(ann, "mask_2d", None)
        if mask is None and isinstance(ann, dict):
            mask = ann.get("mask_2d")
        if mask is None:
            continue
        mask_array = np.asarray(mask)
        if mask_array.ndim >= 2 and mask_array.shape[0] > 1 and mask_array.shape[1] > 1:
            return int(mask_array.shape[1]), int(mask_array.shape[0])
    return None


def _scale_point_cloud(point_cloud, scale_factor: float):
    import open3d as o3d

    points = np.asarray(point_cloud.points, dtype=np.float64) * scale_factor
    colors = np.asarray(point_cloud.colors, dtype=np.float64)
    scaled = o3d.geometry.PointCloud()
    scaled.points = o3d.utility.Vector3dVector(points)
    if colors.shape[0] == points.shape[0]:
        scaled.colors = o3d.utility.Vector3dVector(colors.copy())
    return scaled


def _project_depth_image(point_cloud, command: ScaleObservationCommand):
    cfg = command.advanced_config
    required = (
        cfg.image_width,
        cfg.image_height,
        cfg.fx,
        cfg.fy,
        cfg.cx,
        cfg.cy,
    )
    if None in required:
        raise ValueError(
            "return_scaled_depth requires image_width, image_height, fx, fy, cx, and cy."
        )
    import open3d as o3d

    width = int(cfg.image_width)
    height = int(cfg.image_height)
    depth = np.zeros((height, width), dtype=np.float32)
    points = np.asarray(point_cloud.points, dtype=np.float64)
    if points.size == 0:
        return o3d.geometry.Image(depth)
    z_camera = -points[:, 2]
    valid = z_camera > 0
    points = points[valid]
    z_camera = z_camera[valid]
    u = np.round(cfg.fx * points[:, 0] / z_camera + cfg.cx).astype(np.int32)
    v = np.round(cfg.cy - cfg.fy * points[:, 1] / z_camera).astype(np.int32)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    for uu, vv, zz in zip(u[in_bounds], v[in_bounds], z_camera[in_bounds]):
        current = depth[vv, uu]
        if current == 0.0 or zz < current:
            depth[vv, uu] = zz
    return o3d.geometry.Image(depth)
