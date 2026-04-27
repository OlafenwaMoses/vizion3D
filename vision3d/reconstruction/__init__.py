from vision3d.core.container import command_bus, register_command_handler

from .commands import SfMCommand
from .handlers import SfMHandler
from .models import SfMResult

# Register handlers on import or application startup
register_command_handler(SfMCommand, SfMHandler)


class StructureFromMotion:
    """
    Facade for the Structure from Motion (SfM) task.

    This class serves as the primary entry point for triggering sparse scene
    reconstruction and camera pose estimation via direct Python import.

    Example:
        ```python
        images = {"view_1.jpg": b"...", "view_2.jpg": b"..."}
        cmd = SfMCommand(images=images)
        task = StructureFromMotion()
        result = task.run(cmd)
        ```
    """

    experimental: bool = False

    def run(self, command: SfMCommand) -> SfMResult:
        """
        Dispatches the provided command through the CQRS bus to the registered handler.

        Args:
            command (SfMCommand): The batch of images and inference parameters.

        Returns:
            SfMResult: The resultant point cloud and camera extrinsics.
        """
        return command_bus.dispatch(command)


__all__ = ["StructureFromMotion", "SfMCommand", "SfMResult"]
