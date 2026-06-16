"""Direct Python entry points for 3D reconstruction tasks."""

from vizion3d.core.container import command_bus, register_command_handler

from .commands import (
    Object3DReconstructionCommand,
    SceneComponents3DReconstructionCommand,
)
from .handlers import (
    Object3DReconstructionHandler,
    SceneComponents3DReconstructionHandler,
)
from .models import (
    Object3DReconstructionConfig,
    Object3DReconstructionResult,
    SceneComponent3D,
    SceneComponents3DReconstructionConfig,
    SceneComponents3DReconstructionResult,
)

register_command_handler(Object3DReconstructionCommand, Object3DReconstructionHandler)
register_command_handler(
    SceneComponents3DReconstructionCommand,
    SceneComponents3DReconstructionHandler,
)


class Object3DReconstruction:
    def run(
        self, command: Object3DReconstructionCommand
    ) -> Object3DReconstructionResult:
        return command_bus.dispatch(command)


class SceneComponents3DReconstruction:
    def run(
        self, command: SceneComponents3DReconstructionCommand
    ) -> SceneComponents3DReconstructionResult:
        return command_bus.dispatch(command)


__all__ = [
    "Object3DReconstruction",
    "Object3DReconstructionCommand",
    "Object3DReconstructionConfig",
    "Object3DReconstructionResult",
    "SceneComponent3D",
    "SceneComponents3DReconstruction",
    "SceneComponents3DReconstructionCommand",
    "SceneComponents3DReconstructionConfig",
    "SceneComponents3DReconstructionResult",
]
