"""CQRS command payload for the ScaleObservation task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vizion3d.annotation.models import MaskAnnotation3D
from vizion3d.core.cqrs import Command

from .models import ScaleObservationAdvancedConfig, ScaleObservationConfig, ScaleObservationResult


@dataclass
class ScaleObservationCommand(Command[ScaleObservationResult]):
    """Command payload to estimate metric scale for a generated point cloud.

    Attributes:
        point_cloud: Open3D point cloud, PLY bytes, or a PLY file path.
        annotations: Usually ``ObjectMaskAnnotation3DResult.annotations``.
            JSON-compatible dictionaries are also accepted for REST/gRPC adapters.
        image_input: Optional RGB image path or bytes. The current runtime only
            uses this for future extensibility; annotations are the scale evidence.
        return_scaled_point_cloud: Include a uniformly scaled Open3D point cloud.
        return_scaled_depth: Include a projected camera-space Z depth image when
            intrinsics and image dimensions are provided.
        config: ScaleObservation estimator configuration. Defaults to promoted V4.1.
        advanced_config: Image size and camera intrinsics for bbox edge checks
            and optional scaled-depth reprojection.
    """

    point_cloud: object
    annotations: list[MaskAnnotation3D] | list[dict[str, Any]] | None = None
    image_input: str | bytes | None = None
    return_scaled_point_cloud: bool = False
    return_scaled_depth: bool = False
    return_report: bool = True
    config: ScaleObservationConfig = field(default_factory=ScaleObservationConfig)
    advanced_config: ScaleObservationAdvancedConfig = field(
        default_factory=ScaleObservationAdvancedConfig
    )
