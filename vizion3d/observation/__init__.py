"""ScaleObservation task direct Python entry point."""

from vizion3d.core.container import command_bus, register_command_handler

from .commands import ScaleObservationCommand
from .handlers import ScaleObservationHandler
from .models import (
    ObjectScaleObservation,
    ScaleCandidate,
    ScaledDepthMetadata,
    ScaleObservationAdvancedConfig,
    ScaleObservationConfig,
    ScaleObservationResult,
)

register_command_handler(ScaleObservationCommand, ScaleObservationHandler)


class ScaleObservation:
    """Facade for the ScaleObservation task."""

    experimental: bool = False

    def run(self, command: ScaleObservationCommand) -> ScaleObservationResult:
        """Dispatch *command* through the CQRS bus to the registered handler."""

        return command_bus.dispatch(command)


__all__ = [
    "ObjectScaleObservation",
    "ScaleCandidate",
    "ScaleObservation",
    "ScaleObservationAdvancedConfig",
    "ScaleObservationCommand",
    "ScaleObservationConfig",
    "ScaleObservationResult",
    "ScaledDepthMetadata",
]
