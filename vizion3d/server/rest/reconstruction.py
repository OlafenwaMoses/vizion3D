"""REST endpoints for object and scene-component 3D reconstruction."""

from __future__ import annotations

import orjson
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from vizion3d.reconstruction import (
    Object3DReconstruction,
    Object3DReconstructionCommand,
    Object3DReconstructionConfig,
    SceneComponents3DReconstruction,
    SceneComponents3DReconstructionCommand,
    SceneComponents3DReconstructionConfig,
)

from .serialisation import b64, o3d_point_cloud_to_ply_bytes, trimesh_to_ply_bytes

router = APIRouter()


def _object_config(
    max_input_dimension: int,
    marching_cubes_resolution: int,
    density_threshold: float,
    point_count: int,
    device: str,
    foreground_ratio: float,
    smoothing_iterations: int,
    min_component_area_ratio: float,
) -> Object3DReconstructionConfig:
    return Object3DReconstructionConfig(
        max_input_dimension=max_input_dimension,
        marching_cubes_resolution=marching_cubes_resolution,
        density_threshold=density_threshold,
        point_count=point_count,
        device=device,
        foreground_ratio=foreground_ratio,
        smoothing_iterations=smoothing_iterations,
        min_component_area_ratio=min_component_area_ratio,
    )


@router.post("/object-3d-reconstruction")
async def object_3d_reconstruction(
    image: UploadFile = File(...),
    model_bundle: str | None = Form(None),
    max_input_dimension: int = Form(1080),
    marching_cubes_resolution: int = Form(256),
    density_threshold: float = Form(25.0),
    point_count: int = Form(200_000),
    device: str = Form("auto"),
    foreground_ratio: float = Form(0.82),
    smoothing_iterations: int = Form(5),
    min_component_area_ratio: float = Form(0.02),
):
    result = Object3DReconstruction().run(
        Object3DReconstructionCommand(
            image_input=await image.read(),
            model_bundle=model_bundle,
            advanced_config=_object_config(
                max_input_dimension,
                marching_cubes_resolution,
                density_threshold,
                point_count,
                device,
                foreground_ratio,
                smoothing_iterations,
                min_component_area_ratio,
            ),
        )
    )
    payload = {
        "mesh_ply": b64(trimesh_to_ply_bytes(result.mesh)),
        "point_cloud_ply": b64(o3d_point_cloud_to_ply_bytes(result.point_cloud)),
        "vertex_count": result.vertex_count,
        "face_count": result.face_count,
        "point_count": result.point_count,
        "backend_used": result.backend_used,
    }
    return Response(content=orjson.dumps(payload), media_type="application/json")


@router.post("/scene-components-3d-reconstruction")
async def scene_components_3d_reconstruction(
    image: UploadFile = File(...),
    model_bundle: str | None = Form(None),
    depth_model_backend: str | None = Form(None),
    annotation_model_backend: str | None = Form(None),
    max_input_dimension: int = Form(1080),
    max_objects: int = Form(0),
    confidence_threshold: float = Form(0.25),
    padding_ratio: float = Form(0.15),
    marching_cubes_resolution: int = Form(256),
    density_threshold: float = Form(25.0),
    point_count: int = Form(200_000),
    device: str = Form("auto"),
    foreground_ratio: float = Form(0.82),
    smoothing_iterations: int = Form(5),
    min_component_area_ratio: float = Form(0.02),
):
    result = SceneComponents3DReconstruction().run(
        SceneComponents3DReconstructionCommand(
            image_input=await image.read(),
            model_bundle=model_bundle,
            depth_model_backend=depth_model_backend,
            annotation_model_backend=annotation_model_backend,
            advanced_config=SceneComponents3DReconstructionConfig(
                max_input_dimension=max_input_dimension,
                max_objects=max_objects,
                confidence_threshold=confidence_threshold,
                padding_ratio=padding_ratio,
                object_config=_object_config(
                    1080,
                    marching_cubes_resolution,
                    density_threshold,
                    point_count,
                    device,
                    foreground_ratio,
                    smoothing_iterations,
                    min_component_area_ratio,
                ),
            ),
        )
    )
    payload = {
        "components": [
            {
                "label": component.label,
                "class_id": component.class_id,
                "confidence": component.confidence,
                "bbox_2d": component.bbox_2d,
                "mesh_ply": b64(trimesh_to_ply_bytes(component.mesh)),
                "point_cloud_ply": b64(
                    o3d_point_cloud_to_ply_bytes(component.point_cloud)
                ),
                "vertex_count": component.vertex_count,
                "face_count": component.face_count,
                "point_count": component.point_count,
            }
            for component in result.components
        ],
        "source_image_size": result.source_image_size,
        "analysis_image_size": result.analysis_image_size,
        "depth_backend_used": result.depth_backend_used,
        "annotation_backend_used": result.annotation_backend_used,
        "reconstruction_backend_used": result.reconstruction_backend_used,
    }
    return Response(content=orjson.dumps(payload), media_type="application/json")
