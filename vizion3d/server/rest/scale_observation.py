"""REST endpoint for the ScaleObservation task."""

from __future__ import annotations

import orjson
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from vizion3d.observation import (
    ScaleObservation,
    ScaleObservationAdvancedConfig,
    ScaleObservationCommand,
    ScaleObservationConfig,
)

from .serialisation import b64, o3d_depth_image_to_png_bytes, o3d_point_cloud_to_ply_bytes

router = APIRouter()


@router.post("/scale-observation")
async def scale_observation(
    point_cloud: UploadFile = File(...),
    annotations_json: str | None = Form(None),
    annotations_file: UploadFile | None = File(None),
    return_scaled_point_cloud: bool = Form(False),
    return_scaled_depth: bool = Form(False),
    return_report: bool = Form(True),
    image_width: int | None = Form(None),
    image_height: int | None = Form(None),
    fx: float | None = Form(None),
    fy: float | None = Form(None),
    cx: float | None = Form(None),
    cy: float | None = Form(None),
):
    """Estimate and optionally apply metric scale to a generated point cloud."""

    point_cloud_bytes = await point_cloud.read()
    annotations = None
    if annotations_file is not None:
        annotations = orjson.loads(await annotations_file.read())
    elif annotations_json:
        annotations = orjson.loads(annotations_json)
    cmd = ScaleObservationCommand(
        point_cloud=point_cloud_bytes,
        annotations=annotations,
        return_scaled_point_cloud=return_scaled_point_cloud,
        return_scaled_depth=return_scaled_depth,
        return_report=return_report,
        config=ScaleObservationConfig(),
        advanced_config=ScaleObservationAdvancedConfig(
            image_width=image_width,
            image_height=image_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        ),
    )
    result = ScaleObservation().run(cmd)
    payload = {
        "scale_factor": result.scale_factor,
        "scale_confidence": result.scale_confidence,
        "scale_confidence_reason": result.scale_confidence_reason,
        "algorithm_version": result.algorithm_version,
        "accepted_candidates": result.accepted_candidates,
        "rejected_candidates": result.rejected_candidates,
        "candidates": [c.model_dump() for c in result.candidates],
        "scaled_point_cloud_ply": b64(
            o3d_point_cloud_to_ply_bytes(result.scaled_point_cloud)
            if result.scaled_point_cloud is not None
            else None
        ),
        "scaled_depth_png": b64(
            o3d_depth_image_to_png_bytes(result.scaled_depth_image)
            if result.scaled_depth_image is not None
            else None
        ),
        "scaled_depth_metadata": (
            result.scaled_depth_metadata.model_dump()
            if result.scaled_depth_metadata is not None
            else None
        ),
        "scale_report": result.scale_report,
    }
    return Response(content=orjson.dumps(payload), media_type="application/json")
