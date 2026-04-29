import base64
import io

import numpy as np
import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from vizion3d.lifting import DepthEstimation, DepthEstimationCommand
from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_BACKEND
from vizion3d.lifting.utils import create_mesh_ply_binary, create_ply_binary

_MAX_BODY = 500 * 1024 * 1024   # 500 MB

app = FastAPI(title="vizion3d REST API", version="1.0.0")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_BODY:
        return JSONResponse(
            {"detail": f"Request body exceeds the 500 MB limit."},
            status_code=413,
        )
    return await call_next(request)

lifting_router = APIRouter(prefix="/lifting", tags=["Lifting (2D -> 3D)"])


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

    def _b64(data: bytes | None) -> str | None:
        return base64.b64encode(data).decode() if data is not None else None

    return {
        "depth_map": result.depth_map,
        "min_depth": result.min_depth,
        "max_depth": result.max_depth,
        "backend_used": result.backend_used,
        "depth_image": _b64(
            _o3d_depth_image_to_png_bytes(result.depth_image)
            if result.depth_image is not None
            else None
        ),
        "point_cloud_ply": _b64(
            _o3d_point_cloud_to_ply_bytes(result.point_cloud)
            if result.point_cloud is not None
            else None
        ),
        "mesh_ply": _b64(
            _o3d_mesh_to_ply_bytes(result.mesh) if result.mesh is not None else None
        ),
    }


app.include_router(lifting_router)


def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
