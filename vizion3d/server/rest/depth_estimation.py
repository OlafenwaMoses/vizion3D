"""
REST endpoint for the Depth Estimation feature.

Registers ``POST /lifting/depth-estimation`` on the ``lifting_router`` exported
from this module.  Import the router in ``app.py`` and call
``app.include_router(lifting_router)``.
"""

from fastapi import APIRouter, File, Form, UploadFile

from vizion3d.lifting import DepthEstimation, DepthEstimationCommand
from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_URL
from vizion3d.lifting.models import DepthEstimationAdvanceConfig

from .serialisation import (
    b64,
    o3d_depth_image_to_png_bytes,
    o3d_mesh_to_ply_bytes,
    o3d_point_cloud_to_ply_bytes,
)

router = APIRouter()

# Set by configure_model() at server startup when --depth_model is passed on the CLI.
_model_override: str | None = None


def configure_model(path: str) -> None:
    """Set a server-wide default model path and pre-load it into handler memory.

    Called from ``app.run()`` when ``--depth_model`` is supplied on the command line.
    After this call the endpoint uses *path* whenever the caller omits
    ``model_backend`` from the form data.
    """
    global _model_override
    _model_override = path
    from vizion3d.lifting.handlers import DepthEstimationHandler

    DepthEstimationHandler.preload(path)


@router.post("/depth-estimation")
async def depth_estimation(
    image: UploadFile = File(...),
    model_backend: str | None = Form(None),
    return_depth_image: bool = Form(False),
    return_point_cloud: bool = Form(False),
    return_mesh: bool = Form(False),
    fx: float | None = Form(None),
    fy: float | None = Form(None),
    cx: float | None = Form(None),
    cy: float | None = Form(None),
    depth_scale: float | None = Form(None),
    depth_trunc: float | None = Form(None),
):
    """Run monocular depth estimation on a single uploaded image.

    Args:
        image: The image file (any PIL-supported format).
        model_backend: Checkpoint URL or local path (defaults to the vizion3D release).
        return_depth_image: Include a base64-encoded 16-bit PNG depth image.
        return_point_cloud: Include a base64-encoded binary PLY point cloud.
        return_mesh: Include a base64-encoded binary PLY surface mesh.
        fx, fy, cx, cy: Camera intrinsics (uses PrimeSense defaults if omitted).
        depth_scale: Raw uint16 → metres divisor (default 1000).
        depth_trunc: Maximum depth in metres (default 10).

    Returns:
        JSON with ``depth_map``, ``min_depth``, ``max_depth``, ``backend_used``,
        and optional ``depth_image``, ``point_cloud_ply``, ``mesh_ply`` (base64).
    """
    image_bytes = await image.read()
    effective_backend = model_backend or _model_override or DEFAULT_DEPTH_MODEL_URL
    base_cfg = DepthEstimationAdvanceConfig()
    advanced_config = DepthEstimationAdvanceConfig(
        fx=fx if fx is not None else base_cfg.fx,
        fy=fy if fy is not None else base_cfg.fy,
        cx=cx if cx is not None else base_cfg.cx,
        cy=cy if cy is not None else base_cfg.cy,
        depth_scale=depth_scale if depth_scale is not None else base_cfg.depth_scale,
        depth_trunc=depth_trunc if depth_trunc is not None else base_cfg.depth_trunc,
    )
    cmd = DepthEstimationCommand(
        image_input=image_bytes,
        model_backend=effective_backend,
        return_depth_image=return_depth_image,
        return_point_cloud=return_point_cloud,
        return_mesh=return_mesh,
        advanced_config=advanced_config,
    )
    result = DepthEstimation().run(cmd)
    return {
        "depth_map": result.depth_map,
        "min_depth": result.min_depth,
        "max_depth": result.max_depth,
        "backend_used": result.backend_used,
        "depth_image": b64(
            o3d_depth_image_to_png_bytes(result.depth_image)
            if result.depth_image is not None
            else None
        ),
        "point_cloud_ply": b64(
            o3d_point_cloud_to_ply_bytes(result.point_cloud)
            if result.point_cloud is not None
            else None
        ),
        "mesh_ply": b64(o3d_mesh_to_ply_bytes(result.mesh) if result.mesh is not None else None),
    }
