from vizion3d.core.container import command_bus, register_command_handler

from .commands import ObjectMaskAnnotation3DCommand
from .handlers import ObjectMaskAnnotation3DHandler
from .models import (
    MaskAnnotation3D,
    ObjectMaskAnnotation3DConfig,
    ObjectMaskAnnotation3DResult,
)

register_command_handler(ObjectMaskAnnotation3DCommand, ObjectMaskAnnotation3DHandler)


class ObjectMaskAnnotation3D:
    """Facade for the ObjectMaskAnnotation3D task.

    Detects and instance-segments objects in a 2D image (or a synthesised
    front-view of the point cloud when no image is supplied), then attributes
    each segmentation mask to the matching 3D points via camera back-projection.

    Example — with a real image::

        import open3d as o3d
        from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand

        pcd = o3d.io.read_point_cloud("scene.ply")
        result = ObjectMaskAnnotation3D().run(
            ObjectMaskAnnotation3DCommand(
                point_cloud=pcd,
                image_input="scene.png",
                return_annotated_cloud=True,
            )
        )

    Example — point cloud only (no image)::

        result = ObjectMaskAnnotation3D().run(
            ObjectMaskAnnotation3DCommand(point_cloud=pcd)
        )
    """

    def run(self, command: ObjectMaskAnnotation3DCommand) -> ObjectMaskAnnotation3DResult:
        return command_bus.dispatch(command)


__all__ = [
    "ObjectMaskAnnotation3D",
    "ObjectMaskAnnotation3DCommand",
    "ObjectMaskAnnotation3DConfig",
    "ObjectMaskAnnotation3DResult",
    "MaskAnnotation3D",
]
