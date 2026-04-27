from vision3d.core.container import command_bus, register_command_handler

from .commands import DepthEstimationCommand
from .handlers import DepthEstimationHandler
from .models import DepthEstimationResult

# Register handlers on import or application startup
register_command_handler(DepthEstimationCommand, DepthEstimationHandler)


class DepthEstimation:
    """
    Facade for the Depth Estimation task.

    This class serves as the primary entry point for triggering monocular depth
    estimation inference via direct Python import.

    Example:
        ```python
        cmd = DepthEstimationCommand(image_input=b"...", return_mesh=True)
        task = DepthEstimation()
        result = task.run(cmd)
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


__all__ = ["DepthEstimation", "DepthEstimationCommand", "DepthEstimationResult"]
