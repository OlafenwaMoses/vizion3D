"""
REST endpoint for the Stereo Depth feature.

Registers ``POST /lifting/stereo-depth`` on the ``router`` exported from this
module.  Import the router in ``app.py`` and call
``app.include_router(router)``.
"""

from fastapi import APIRouter, File, Form, UploadFile

from vizion3d.stereo import StereoDepth, StereoDepthCommand
from vizion3d.stereo.defaults import DEFAULT_STEREO_MODEL_URL
from vizion3d.stereo.models import StereoDepthAdvancedConfig

from .serialisation import (
    b64,
    o3d_depth_image_to_png_bytes,
    o3d_mesh_to_ply_bytes,
    o3d_point_cloud_to_ply_bytes,
)

router = APIRouter()

# Set by configure_model() at server startup when --stereo_model is passed on the CLI.
_model_override: str | None = None


def configure_model(path: str) -> None:
    """Set a server-wide default stereo model path and pre-load it into handler memory.

    Called from ``app.run()`` when ``--stereo_model`` is supplied on the command line.
    After this call the endpoint uses *path* whenever the caller omits
    ``model_backend`` from the form data.
    """
    global _model_override
    _model_override = path
    from vizion3d.stereo.handlers import StereoDepthHandler

    StereoDepthHandler.preload(path)


@router.post("/stereo-depth")
async def stereo_depth(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    model_backend: str | None = Form(None),
    return_depth_image: bool = Form(False),
    return_point_cloud: bool = Form(False),
    return_mesh: bool = Form(False),
    focal_length: float | None = Form(None),
    cx: float | None = Form(None),
    cy: float | None = Form(None),
    baseline: float | None = Form(None),
    doffs: float | None = Form(None),
    z_far: float | None = Form(None),
    conf_threshold: float | None = Form(None),
    occ_threshold: float | None = Form(None),
    scale_factor: float | None = Form(None),
):
    """Run stereo depth estimation on a rectified left/right image pair.

    Args:
        left_image: Left-camera image file (any PIL-supported format).
        right_image: Right-camera image file (same resolution, horizontally offset).
        model_backend: S2M2 checkpoint URL or local path (defaults to the vizion3D release).
        return_depth_image: Include a base64-encoded 16-bit PNG depth image.
        return_point_cloud: Include a base64-encoded binary PLY point cloud.
        return_mesh: Include a base64-encoded binary PLY surface mesh.
        focal_length: Focal length in pixels (default 1000.0 — override with your calibration).
        cx, cy: Principal point in pixels.
        baseline: Stereo baseline in millimetres (default 100.0).
        doffs: Disparity offset (default 0.0).
        z_far: Max depth in metres for point cloud (default 10.0).
        conf_threshold: Minimum confidence for point inclusion (default 0.1).
        occ_threshold: Minimum occlusion score for point inclusion (default 0.5).
        scale_factor: Input downscale factor for speed/quality tradeoff (default 1.0).

    Returns:
        JSON with ``depth_map``, ``disparity_map``, ``min_depth``, ``max_depth``,
        ``backend_used``, and optional ``depth_image``, ``point_cloud_ply``,
        ``mesh_ply`` (base64).
    """
    left_bytes = await left_image.read()
    right_bytes = await right_image.read()
    effective_backend = model_backend or _model_override or DEFAULT_STEREO_MODEL_URL

    base_cfg = StereoDepthAdvancedConfig()
    advanced_config = StereoDepthAdvancedConfig(
        focal_length=focal_length if focal_length is not None else base_cfg.focal_length,
        cx=cx if cx is not None else base_cfg.cx,
        cy=cy if cy is not None else base_cfg.cy,
        baseline=baseline if baseline is not None else base_cfg.baseline,
        doffs=doffs if doffs is not None else base_cfg.doffs,
        z_far=z_far if z_far is not None else base_cfg.z_far,
        conf_threshold=conf_threshold if conf_threshold is not None else base_cfg.conf_threshold,
        occ_threshold=occ_threshold if occ_threshold is not None else base_cfg.occ_threshold,
        scale_factor=scale_factor if scale_factor is not None else base_cfg.scale_factor,
    )
    cmd = StereoDepthCommand(
        left_image=left_bytes,
        right_image=right_bytes,
        model_backend=effective_backend,
        return_depth_image=return_depth_image,
        return_point_cloud=return_point_cloud,
        return_mesh=return_mesh,
        advanced_config=advanced_config,
    )
    result = StereoDepth().run(cmd)
    return {
        "depth_map": result.depth_map,
        "disparity_map": result.disparity_map,
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
