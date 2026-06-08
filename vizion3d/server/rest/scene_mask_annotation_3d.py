"""
REST endpoint for the SceneMaskAnnotation3D feature.

Registers ``POST /annotation/scene-mask-annotation-3d`` on the ``router``
exported from this module.  Import the router in ``app.py`` and include it on
the annotation router.

Input
-----
- ``point_cloud_ply``  — binary PLY file of the point cloud (required)
- ``image``            — RGB image file (optional; synthesised from cloud if omitted)
- Optional form fields for config overrides and feature flags

Output
------
JSON payload with one entry per semantic class present in the scene.  Binary
fields (masks, per-class PLY clouds, annotated cloud) are base64-encoded inside
the JSON.
"""

import orjson
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from vizion3d.annotation import SceneMaskAnnotation3D, SceneMaskAnnotation3DCommand
from vizion3d.annotation.models import SceneMaskAnnotation3DConfig
from vizion3d.annotation.scene_defaults import DEFAULT_SCENE_MODEL_URL

from .serialisation import (
    b64,
    mask_to_png_bytes,
    o3d_point_cloud_to_ply_bytes,
    ply_bytes_to_o3d_point_cloud,
)

router = APIRouter()

_model_override: str | None = None


def configure_model(path: str) -> None:
    """Set a server-wide default scene model path and pre-load it."""
    global _model_override
    _model_override = path
    from vizion3d.annotation.scene_handlers import SceneMaskAnnotation3DHandler

    SceneMaskAnnotation3DHandler.preload(path)


@router.post("/scene-mask-annotation-3d")
async def scene_mask_annotation_3d(
    point_cloud_ply: UploadFile = File(...),
    image: UploadFile | None = File(None),
    model_backend: str | None = Form(None),
    return_region_clouds: bool = Form(False),
    return_annotated_cloud: bool = Form(False),
    fx: float | None = Form(None),
    fy: float | None = Form(None),
    cx: float | None = Form(None),
    cy: float | None = Form(None),
    inference_size: int | None = Form(None),
):
    """Semantically segment a scene and group point-cloud points by class.

    Args:
        point_cloud_ply: Binary PLY file of the point cloud.
        image: Optional RGB image.  When omitted, a synthetic front-view image
            is rendered from the point cloud XYZ+RGB using the camera
            intrinsics in the config.
        model_backend: SegFormer-B4 checkpoint URL or local path.
        return_region_clouds: Include a base64 PLY per semantic class.
        return_annotated_cloud: Include a base64 PLY of the full cloud recoloured
            by each point's class palette colour.
        fx, fy, cx, cy: Camera intrinsics for back-projection.
        inference_size: Shorter-edge size for the network (0 = native).

    Returns:
        JSON with an ``annotations`` list, ``annotated_cloud_ply``, and
        ``backend_used``.
    """
    ply_bytes = await point_cloud_ply.read()
    image_bytes = await image.read() if image is not None else None

    effective_backend = model_backend or _model_override or DEFAULT_SCENE_MODEL_URL
    point_cloud = ply_bytes_to_o3d_point_cloud(ply_bytes)

    base_cfg = SceneMaskAnnotation3DConfig()
    advanced_config = SceneMaskAnnotation3DConfig(
        fx=fx if fx is not None else base_cfg.fx,
        fy=fy if fy is not None else base_cfg.fy,
        cx=cx if cx is not None else base_cfg.cx,
        cy=cy if cy is not None else base_cfg.cy,
        inference_size=inference_size if inference_size is not None else base_cfg.inference_size,
    )

    cmd = SceneMaskAnnotation3DCommand(
        point_cloud=point_cloud,
        image_input=image_bytes,
        model_backend=effective_backend,
        return_region_clouds=return_region_clouds,
        return_annotated_cloud=return_annotated_cloud,
        advanced_config=advanced_config,
    )
    result = SceneMaskAnnotation3D().run(cmd)

    annotations_out = []
    for ann in result.annotations:
        annotations_out.append(
            {
                "label": ann.label,
                "class_id": ann.class_id,
                "bbox_2d": ann.bbox_2d,
                "pixel_count": ann.pixel_count,
                "mask_image": b64(mask_to_png_bytes(ann.mask_2d)),
                "point_indices": ann.point_indices,
                "point_coords": ann.point_coords,
                "region_cloud_ply": b64(
                    o3d_point_cloud_to_ply_bytes(ann.region_cloud)
                    if ann.region_cloud is not None
                    else None
                ),
            }
        )

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
