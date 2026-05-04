from vizion3d.core.container import command_bus, register_command_handler

from .commands import DepthEstimationCommand
from .handlers import DepthEstimationHandler
from .models import DepthEstimationAdvanceConfig, DepthEstimationResult

# Register handlers on import or application startup
register_command_handler(DepthEstimationCommand, DepthEstimationHandler)


class DepthEstimation:
    """
    Facade for the Depth Estimation task.

    This class serves as the primary entry point for triggering monocular depth
    estimation inference via direct Python import.

    Example:
        ```python
        from vizion3d.lifting import (
            DepthEstimation,
            DepthEstimationAdvanceConfig,
            DepthEstimationCommand,
        )

        cmd = DepthEstimationCommand(
            image_input=b"...",
            return_point_cloud=True,
            return_mesh=True,
            advanced_config=DepthEstimationAdvanceConfig(
                fx=615.0, fy=615.0, cx=320.0, cy=240.0, depth_trunc=5.0
            ),
        )
        result = DepthEstimation().run(cmd)
        ```
    """

    experimental: bool = False

    def run(self, command: DepthEstimationCommand) -> DepthEstimationResult:
        """
        Dispatches the provided command through the CQRS bus to the registered handler.

        Args:
            command (DepthEstimationCommand): The inference parameters and flags.

        Returns:
            DepthEstimationResult: The resultant depth map and optional generated files.
        """
        return command_bus.dispatch(command)


__all__ = [
    "DepthEstimation",
    "DepthEstimationAdvanceConfig",
    "DepthEstimationCommand",
    "DepthEstimationResult",
]
