"""
REST endpoint for the ObjectMaskAnnotation3D feature.

Registers ``POST /annotation/object-mask-annotation-3d`` on the ``router``
exported from this module.  Import the router in ``app.py`` and include it on
the annotation router.

Input
-----
- ``point_cloud_ply``  — binary PLY file of the point cloud (required)
- ``image``            — RGB image file (optional; synthesised from cloud if omitted)
- Optional form fields for config overrides and feature flags

Output
------
JSON payload with per-object mask annotations.  Binary fields (masks,
per-object PLY clouds, annotated cloud) are base64-encoded inside the JSON.
"""

import orjson
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand
from vizion3d.annotation.defaults import DEFAULT_ANNOTATION_MODEL_URL
from vizion3d.annotation.models import ObjectMaskAnnotation3DConfig

from .serialisation import (
    b64,
    mask_to_png_bytes,
    o3d_point_cloud_to_ply_bytes,
    ply_bytes_to_o3d_point_cloud,
)

router = APIRouter()

_model_override: str | None = None


def configure_model(path: str) -> None:
    """Set a server-wide default annotation model path and pre-load it."""
    global _model_override
    _model_override = path
    from vizion3d.annotation.handlers import ObjectMaskAnnotation3DHandler

    ObjectMaskAnnotation3DHandler.preload(path)


@router.post("/object-mask-annotation-3d")
async def object_mask_annotation_3d(
    point_cloud_ply: UploadFile = File(...),
    image: UploadFile | None = File(None),
    model_backend: str | None = Form(None),
    return_object_clouds: bool = Form(False),
    return_annotated_cloud: bool = Form(False),
    fx: float | None = Form(None),
    fy: float | None = Form(None),
    cx: float | None = Form(None),
    cy: float | None = Form(None),
    conf_threshold: float | None = Form(None),
    iou_threshold: float | None = Form(None),
):
    """Detect, segment, and mask-annotate objects in a point cloud.

    Args:
        point_cloud_ply: Binary PLY file of the point cloud.
        image: Optional RGB image.  When omitted, a synthetic front-view image
            is rendered from the point cloud XYZ+RGB using the camera
            intrinsics in the config.
        model_backend: YOLO11n-seg checkpoint URL or local path.
        return_object_clouds: Include a base64 PLY per detected object.
        return_annotated_cloud: Include a base64 PLY of the full cloud with
            object colours painted on.
        fx, fy, cx, cy: Camera intrinsics for back-projection.
        conf_threshold: Minimum detection confidence.
        iou_threshold: NMS IoU threshold.

    Returns:
        JSON with an ``annotations`` list, ``annotated_cloud_ply``, and
        ``backend_used``.
    """
    ply_bytes = await point_cloud_ply.read()
    image_bytes = await image.read() if image is not None else None

    effective_backend = model_backend or _model_override or DEFAULT_ANNOTATION_MODEL_URL
    point_cloud = ply_bytes_to_o3d_point_cloud(ply_bytes)

    base_cfg = ObjectMaskAnnotation3DConfig()
    advanced_config = ObjectMaskAnnotation3DConfig(
        fx=fx if fx is not None else base_cfg.fx,
        fy=fy if fy is not None else base_cfg.fy,
        cx=cx if cx is not None else base_cfg.cx,
        cy=cy if cy is not None else base_cfg.cy,
        conf_threshold=conf_threshold if conf_threshold is not None else base_cfg.conf_threshold,
        iou_threshold=iou_threshold if iou_threshold is not None else base_cfg.iou_threshold,
    )

    cmd = ObjectMaskAnnotation3DCommand(
        point_cloud=point_cloud,
        image_input=image_bytes,
        model_backend=effective_backend,
        return_object_clouds=return_object_clouds,
        return_annotated_cloud=return_annotated_cloud,
        advanced_config=advanced_config,
    )
    result = ObjectMaskAnnotation3D().run(cmd)

    annotations_out = []
    for ann in result.annotations:
        ann_dict = {
            "label": ann.label,
            "class_id": ann.class_id,
            "confidence": ann.confidence,
            "bbox_2d": ann.bbox_2d,
            "mask_image": b64(mask_to_png_bytes(ann.mask_2d)),
            "point_indices": ann.point_indices,
            "point_coords": ann.point_coords,
            "object_cloud_ply": b64(
                o3d_point_cloud_to_ply_bytes(ann.object_cloud)
                if ann.object_cloud is not None
                else None
            ),
        }
        annotations_out.append(ann_dict)

    payload = {
        "annotations": annotations_out,
        "annotated_cloud_ply": b64(
            o3d_point_cloud_to_ply_bytes(result.annotated_cloud)
            if result.annotated_cloud is not None
            else None
        ),
        "backend_used": result.backend_used,
    }
    return Response(content=orjson.dumps(payload), media_type="application/json")
