"""Data models for object and scene-component 3D reconstruction."""

from __future__ import annotations

from typing import Any

from open3d.geometry import PointCloud as O3dPointCloud  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field


class Object3DReconstructionConfig(BaseModel):
    max_input_dimension: int = Field(default=1080, ge=1, le=1080)
    marching_cubes_resolution: int = Field(default=256, ge=32, le=1024)
    density_threshold: float = Field(default=25.0, gt=0)
    point_count: int = Field(default=200_000, ge=1)
    foreground_ratio: float = Field(default=0.82, gt=0, le=1)
    smoothing_iterations: int = Field(default=5, ge=0)
    min_component_area_ratio: float = Field(default=0.02, ge=0, le=1)
    device: str = "auto"


class Object3DReconstructionResult(BaseModel):
    mesh: Any
    point_cloud: O3dPointCloud
    backend_used: str
    vertex_count: int
    face_count: int
    point_count: int
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SceneComponents3DReconstructionConfig(BaseModel):
    max_input_dimension: int = Field(default=1080, ge=0)
    max_objects: int = Field(default=0, ge=0)
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    padding_ratio: float = Field(default=0.15, ge=0)
    object_config: Object3DReconstructionConfig = Field(
        default_factory=Object3DReconstructionConfig
    )


class SceneComponent3D(BaseModel):
    label: str
    class_id: int
    confidence: float
    bbox_2d: list[float]
    mesh: Any
    point_cloud: O3dPointCloud
    vertex_count: int
    face_count: int
    point_count: int
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SceneComponents3DReconstructionResult(BaseModel):
    components: list[SceneComponent3D]
    source_image_size: tuple[int, int]
    analysis_image_size: tuple[int, int]
    depth_backend_used: str
    annotation_backend_used: str
    reconstruction_backend_used: str
    model_config = ConfigDict(arbitrary_types_allowed=True)
