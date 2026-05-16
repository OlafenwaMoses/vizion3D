"""
CQRS command payload for the ObjectMaskAnnotation3D task.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vizion3d.core.cqrs import Command

from .defaults import DEFAULT_ANNOTATION_MODEL_URL
from .models import ObjectMaskAnnotation3DConfig, ObjectMaskAnnotation3DResult


@dataclass
class ObjectMaskAnnotation3DCommand(Command[ObjectMaskAnnotation3DResult]):
    """Command payload to trigger a 3D mask annotation inference task.

    Detects and instance-segments objects in the 2D image, then attributes
    each detected mask to the matching subset of 3D points in the point cloud
    via camera back-projection.

    Attributes:
        point_cloud: An ``open3d.geometry.PointCloud`` in OpenGL/viewer
            camera space (X right, Y up, Z negative forward), coordinates in
            metres.
        image_input: RGB image used for segmentation.  Pass a file-path string
            or raw image bytes.  If ``None``, the handler synthesises a
            front-view RGB image by projecting the point cloud XYZ+RGB into
            2D using the camera intrinsics — no real photo is needed.
        model_backend: YOLO11n-seg checkpoint URL or local path.  Defaults to
            the vizion3D release checkpoint, downloaded on first use and cached
            under ``~/.cache/vizion3d/models/``.  Set ``VIZION3D_MODEL_CACHE``
            to override the cache directory.
        return_object_clouds: When ``True``, each
            :class:`~vizion3d.annotation.models.MaskAnnotation3D` includes an
            extracted ``open3d.geometry.PointCloud`` with original colours.
        return_annotated_cloud: When ``True``, the result includes a copy of
            the full point cloud with detected object points recoloured.
        advanced_config: Camera intrinsics and detection thresholds.
    """

    point_cloud: object  # open3d.geometry.PointCloud
    image_input: str | bytes | None = None
    model_backend: str = DEFAULT_ANNOTATION_MODEL_URL
    return_object_clouds: bool = False
    return_annotated_cloud: bool = False
    advanced_config: ObjectMaskAnnotation3DConfig = field(
        default_factory=ObjectMaskAnnotation3DConfig
    )
