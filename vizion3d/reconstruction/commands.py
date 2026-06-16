"""CQRS commands for 3D reconstruction tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from vizion3d.core.cqrs import Command

from .models import (
    Object3DReconstructionConfig,
    Object3DReconstructionResult,
    SceneComponents3DReconstructionConfig,
    SceneComponents3DReconstructionResult,
)


@dataclass
class Object3DReconstructionCommand(Command[Object3DReconstructionResult]):
    image_input: str | bytes
    model_bundle: str | None = None
    advanced_config: Object3DReconstructionConfig = field(
        default_factory=Object3DReconstructionConfig
    )


@dataclass
class SceneComponents3DReconstructionCommand(
    Command[SceneComponents3DReconstructionResult]
):
    image_input: str | bytes
    model_bundle: str | None = None
    depth_model_backend: str | None = None
    annotation_model_backend: str | None = None
    advanced_config: SceneComponents3DReconstructionConfig = field(
        default_factory=SceneComponents3DReconstructionConfig
    )
