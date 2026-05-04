"""
Stereo Depth task — direct Python entry point.

Import :class:`StereoDepth` and run it with a :class:`StereoDepthCommand` to
obtain metric depth maps, disparity maps, point clouds, and meshes from
rectified left/right stereo image pairs.

Example::

    from vizion3d.stereo import (
        StereoDepth,
        StereoDepthAdvancedConfig,
        StereoDepthCommand,
    )

    cmd = StereoDepthCommand(
        left_image="left.png",
        right_image="right.png",
        return_point_cloud=True,
        return_mesh=True,
        advanced_config=StereoDepthAdvancedConfig(
            focal_length=1733.74,
            cx=792.27,
            cy=541.89,
            baseline=536.62,
        ),
    )
    result = StereoDepth().run(cmd)
    print(f"Depth range: {result.min_depth:.2f} – {result.max_depth:.2f} m")
"""

from vizion3d.core.container import command_bus, register_command_handler

from .commands import StereoDepthCommand
from .handlers import StereoDepthHandler
from .models import StereoDepthAdvancedConfig, StereoDepthResult

register_command_handler(StereoDepthCommand, StereoDepthHandler)


class StereoDepth:
    """Facade for the Stereo Depth task.

    Serves as the primary entry point for stereo depth inference via direct
    Python import.  Internally dispatches through the CQRS command bus to
    :class:`~vizion3d.stereo.handlers.StereoDepthHandler`.

    Example::

        from vizion3d.stereo import StereoDepth, StereoDepthCommand

        cmd = StereoDepthCommand(left_image=b"...", right_image=b"...")
        result = StereoDepth().run(cmd)
    """

    def run(self, command: StereoDepthCommand) -> StereoDepthResult:
        """Dispatch *command* through the CQRS bus to the registered handler.

        Args:
            command: The stereo inference parameters and flags.

        Returns:
            :class:`StereoDepthResult` with metric depth, disparity, and optional
            depth image, point cloud, and mesh.
        """
        return command_bus.dispatch(command)


__all__ = [
    "StereoDepth",
    "StereoDepthAdvancedConfig",
    "StereoDepthCommand",
    "StereoDepthResult",
]
