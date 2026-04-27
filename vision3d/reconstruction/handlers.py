from vision3d.core.cqrs import CommandHandler

from .commands import SfMCommand
from .models import CameraPose, Point3D, SfMResult


class SfMHandler(CommandHandler[SfMCommand, SfMResult]):
    def handle(self, command: SfMCommand) -> SfMResult:
        # Mocking the inference backend for v1 slice

        mock_points = [
            Point3D(x=1.0, y=2.0, z=3.0, r=255, g=0, b=0),
            Point3D(x=-1.0, y=0.5, z=2.0, r=0, g=255, b=0),
        ]

        mock_poses = []
        for img_id in command.images.keys():
            mock_poses.append(
                CameraPose(
                    image_id=img_id,
                    rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    translation_vector=[0.0, 0.0, 0.0],
                )
            )

        return SfMResult(
            sparse_point_cloud=mock_points,
            camera_poses=mock_poses,
            backend_used=command.model_backend + "-mock",
        )
