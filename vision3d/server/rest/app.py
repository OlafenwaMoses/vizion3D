import io
from typing import List

import numpy as np
import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, UploadFile
from PIL import Image

from vision3d.lifting import DepthEstimation, DepthEstimationCommand
from vision3d.lifting.defaults import DEFAULT_DEPTH_MODEL_BACKEND
from vision3d.lifting.utils import create_mesh_ply_binary, create_ply_binary
from vision3d.reconstruction import SfMCommand, StructureFromMotion

app = FastAPI(title="vision3d REST API", version="1.0.0")

lifting_router = APIRouter(prefix="/lifting", tags=["Lifting (2D -> 3D)"])
reconstruction_router = APIRouter(prefix="/reconstruction", tags=["Reconstruction"])


def _o3d_depth_image_to_png_bytes(o3d_image) -> bytes:
    arr = np.asarray(o3d_image)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _o3d_point_cloud_to_ply_bytes(pcd) -> bytes:
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return create_ply_binary(points, colors)


def _o3d_mesh_to_ply_bytes(mesh) -> bytes:
    points = np.asarray(mesh.vertices).astype(np.float32)
    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    return create_mesh_ply_binary(points, colors, faces)


@lifting_router.post("/depth-estimation")
async def depth_estimation(
    image: UploadFile = File(...),
    model_backend: str = Form(DEFAULT_DEPTH_MODEL_BACKEND),
    return_depth_image: bool = Form(False),
    return_point_cloud: bool = Form(False),
    return_mesh: bool = Form(False),
):
    image_bytes = await image.read()
    cmd = DepthEstimationCommand(
        image_input=image_bytes,
        model_backend=model_backend,
        return_depth_image=return_depth_image,
        return_point_cloud=return_point_cloud,
        return_mesh=return_mesh,
    )

    result = DepthEstimation().run(cmd)

    response = {
        "depth_map": result.depth_map,
        "min_depth": result.min_depth,
        "max_depth": result.max_depth,
        "backend_used": result.backend_used,
        "depth_image": (
            _o3d_depth_image_to_png_bytes(result.depth_image)
            if result.depth_image is not None
            else None
        ),
        "point_cloud_ply": (
            _o3d_point_cloud_to_ply_bytes(result.point_cloud)
            if result.point_cloud is not None
            else None
        ),
        "mesh_ply": (
            _o3d_mesh_to_ply_bytes(result.mesh) if result.mesh is not None else None
        ),
    }
    return response


@reconstruction_router.post("/sfm")
async def sfm(images: List[UploadFile] = File(...), model_backend: str = "colmap"):
    image_dict = {}
    for img in images:
        image_dict[img.filename] = await img.read()

    cmd = SfMCommand(images=image_dict, model_backend=model_backend)
    result = StructureFromMotion().run(cmd)

    return result.model_dump()


app.include_router(lifting_router)
app.include_router(reconstruction_router)


def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
