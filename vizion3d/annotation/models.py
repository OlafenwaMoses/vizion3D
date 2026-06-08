"""
Data models for the ObjectMaskAnnotation3D task.
"""

from __future__ import annotations

import numpy as np
from open3d.geometry import PointCloud as O3dPointCloud  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict


class ObjectMaskAnnotation3DConfig(BaseModel):
    """Camera intrinsics and inference settings for 3D mask annotation.

    Attributes:
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Principal point x — pixel column of the optical axis.
        cy: Principal point y — pixel row of the optical axis.
        conf_threshold: Minimum detection confidence. Range ``[0, 1]``.
        iou_threshold: NMS IoU overlap threshold. Range ``[0, 1]``.
    """

    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45


class MaskAnnotation3D(BaseModel):
    """Per-object mask annotation produced by the ObjectMaskAnnotation3D task.

    Each entry corresponds to one detected instance: a 2D bounding box, a
    pixel-level segmentation mask, and the exact subset of 3D points that
    back-project into that mask.

    Attributes:
        label: COCO class name, e.g. ``"person"``.
        class_id: COCO integer class index (0-based).
        confidence: Detection confidence from the segmentation model.
        bbox_2d: Bounding box in image pixels: ``[x1, y1, x2, y2]``.
        mask_2d: Boolean segmentation mask, shape ``(H, W)``.
        point_indices: Indices into the original input point cloud for every
            point that back-projects inside this mask.
        point_coords: ``[[x, y, z], ...]`` in metres for each matched point.
        object_cloud: Extracted ``open3d.geometry.PointCloud`` for this
            object's points with original colours preserved.  Present when
            ``return_object_clouds=True``; ``None`` otherwise.
    """

    label: str
    class_id: int
    confidence: float
    bbox_2d: list[float]
    mask_2d: np.ndarray
    point_indices: list[int]
    point_coords: list[list[float]]
    object_cloud: O3dPointCloud | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ObjectMaskAnnotation3DResult(BaseModel):
    """Result payload for an ObjectMaskAnnotation3D inference task.

    Attributes:
        annotations: Per-object mask annotations in descending confidence order.
        annotated_cloud: Full point cloud with each detected object's points
            recoloured to a unique fixed colour.  Non-object points keep their
            original colour.  Present when ``return_annotated_cloud=True``.
        backend_used: Resolved local file path of the YOLO checkpoint used.
    """

    annotations: list[MaskAnnotation3D]
    annotated_cloud: O3dPointCloud | None = None
    backend_used: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ── SceneMaskAnnotation3D (semantic) ────────────────────────────────────────────


class SceneMaskAnnotation3DConfig(BaseModel):
    """Camera intrinsics and inference settings for semantic scene annotation.

    Attributes:
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Principal point x — pixel column of the optical axis.
        cy: Principal point y — pixel row of the optical axis.
        inference_size: Shorter-image-edge size fed to the network (aspect ratio
            preserved).  Outputs are always at the original image resolution.
            Set to ``0`` to run at native resolution.
    """

    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    inference_size: int = 512


class SemanticMaskAnnotation3D(BaseModel):
    """Per-class semantic mask annotation produced by SceneMaskAnnotation3D.

    Each entry corresponds to one ADE20K *class* present in the scene — not one
    object instance.  All points whose pixel falls in that class's region are
    grouped into a single annotation.

    Attributes:
        label: ADE20K class name (COCO-aligned where a 1:1 synonym exists).
        class_id: ADE20K class index (0-based, ``0..149``).
        bbox_2d: Axis-aligned bounding box of the class region in image pixels
            ``[x1, y1, x2, y2]`` (empty when the class has no pixels).
        mask_2d: Boolean semantic mask for this class, shape ``(H, W)``.
        pixel_count: Number of pixels classified as this class.
        point_indices: Indices into the original point cloud for every point
            that back-projects inside this class's region.
        point_coords: ``[[x, y, z], ...]`` in metres for each matched point.
        region_cloud: Extracted ``open3d.geometry.PointCloud`` for this class's
            points with original colours.  Present when
            ``return_region_clouds=True``; ``None`` otherwise.
    """

    label: str
    class_id: int
    bbox_2d: list[float]
    mask_2d: np.ndarray
    pixel_count: int
    point_indices: list[int]
    point_coords: list[list[float]]
    region_cloud: O3dPointCloud | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SceneMaskAnnotation3DResult(BaseModel):
    """Result payload for a SceneMaskAnnotation3D inference task.

    Attributes:
        annotations: Per-class semantic mask annotations, ordered by descending
            pixel count.
        annotated_cloud: Full point cloud with each point recoloured by its
            class's fixed ADE20K palette colour.  Present when
            ``return_annotated_cloud=True``.
        backend_used: Resolved local file path of the SegFormer checkpoint used.
    """

    annotations: list[SemanticMaskAnnotation3D]
    annotated_cloud: O3dPointCloud | None = None
    backend_used: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
