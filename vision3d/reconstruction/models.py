from typing import List

from pydantic import BaseModel, ConfigDict


class Point3D(BaseModel):
    """
    Represents a single 3D vertex with color data.

    Attributes:
        x: The global X coordinate.
        y: The global Y coordinate.
        z: The global Z coordinate.
        r: Red color channel (0-255).
        g: Green color channel (0-255).
        b: Blue color channel (0-255).
    """

    x: float
    y: float
    z: float
    r: int
    g: int
    b: int


class CameraPose(BaseModel):
    """
    Represents the estimated extrinsics for an input image in the global space.

    Attributes:
        image_id: The string identifier that was passed in with the `SfMCommand`.
        rotation_matrix: A 3x3 array containing the Euler rotation vector.
        translation_vector: A 1x3 array containing the absolute translation in the scene.
    """

    image_id: str
    rotation_matrix: List[List[float]]
    translation_vector: List[float]


class SfMResult(BaseModel):
    """
    Result payload returned after a batch SfM reconstruction task.

    Attributes:
        sparse_point_cloud: A list of the triangulated 3D points forming the scene.
        camera_poses: The resolved camera poses mapping back to the input image IDs.
        backend_used: The backend tool that resolved the scene.
    """

    sparse_point_cloud: List[Point3D]
    camera_poses: List[CameraPose]
    backend_used: str

    model_config = ConfigDict(from_attributes=True)
