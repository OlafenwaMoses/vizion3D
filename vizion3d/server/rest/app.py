"""
FastAPI REST server for the vizion3d service.

Lifting endpoints (prefix /lifting):
- POST /lifting/depth-estimation   — monocular depth from a single image.
- POST /lifting/stereo-depth       — metric depth from a rectified stereo pair.

Annotation endpoints (prefix /annotation):
- POST /annotation/object-mask-annotation-3d — detect, instance-segment, and
  attribute objects in an image + point cloud pair (image optional).

Start with::

    uv run vizion3d-serve-rest

    # Enable only depth estimation:
    uv run vizion3d-serve-rest --depth_estimation

    # All features with custom model paths (pre-loaded at startup):
    uv run vizion3d-serve-rest \\
        --depth_model /path/to/depth_anything_v2_vitb.pth \\
        --stereo_model /path/to/stereo-depth-s2m2-L.pth \\
        --annotation_model /path/to/yolo11n-seg.pt
"""

import argparse

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vizion3d.server.rest import depth_estimation, object_mask_annotation_3d, stereo_depth

_MAX_BODY = 500 * 1024 * 1024  # 500 MB


def create_app(
    *,
    enable_depth_estimation: bool = True,
    enable_stereo_depth: bool = True,
    enable_object_mask_annotation_3d: bool = True,
) -> FastAPI:
    """Build and return a FastAPI application with the selected feature routers.

    Args:
        enable_depth_estimation: Register ``POST /lifting/depth-estimation``.
        enable_stereo_depth: Register ``POST /lifting/stereo-depth``.
        enable_object_mask_annotation_3d: Register
            ``POST /annotation/object-mask-annotation-3d``.

    Returns:
        A fully configured :class:`FastAPI` instance.
    """
    _app = FastAPI(title="vizion3d REST API", version="1.0.0")

    @_app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY:
            return JSONResponse(
                {"detail": "Request body exceeds the 500 MB limit."},
                status_code=413,
            )
        return await call_next(request)

    lifting_router = APIRouter(prefix="/lifting", tags=["Lifting (2D -> 3D)"])
    if enable_depth_estimation:
        lifting_router.include_router(depth_estimation.router)
    if enable_stereo_depth:
        lifting_router.include_router(stereo_depth.router)
    _app.include_router(lifting_router)

    annotation_router = APIRouter(prefix="/annotation", tags=["Annotation"])
    if enable_object_mask_annotation_3d:
        annotation_router.include_router(object_mask_annotation_3d.router)
    _app.include_router(annotation_router)

    return _app


# Module-level app with all features enabled — used by tests and direct imports.
app = create_app()


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vizion3d-serve-rest",
        description="Start the vizion3d REST API server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
feature/model flags (omit all to enable all features):
  --depth_estimation              enable POST /lifting/depth-estimation
  --stereo_depth                  enable POST /lifting/stereo-depth
  --object_mask_annotation_3d     enable POST /annotation/object-mask-annotation-3d

model pre-loading (also enables that feature):
  --depth_model PATH        use PATH as the default depth-estimation model
  --stereo_model PATH       use PATH as the default stereo-depth model
  --annotation_model PATH   use PATH as the default annotation model
""",
    )
    parser.add_argument("--host", default="0.0.0.0", metavar="HOST")
    parser.add_argument("--port", type=int, default=8000, metavar="PORT")
    parser.add_argument("--depth_model", default=None, metavar="PATH")
    parser.add_argument("--stereo_model", default=None, metavar="PATH")
    parser.add_argument("--annotation_model", default=None, metavar="PATH")
    parser.add_argument("--depth_estimation", action="store_true")
    parser.add_argument("--stereo_depth", action="store_true")
    parser.add_argument("--object_mask_annotation_3d", action="store_true")
    return parser.parse_args(argv)


def run(argv=None) -> None:
    args = _parse_args(argv)

    any_selector = (
        args.depth_estimation
        or args.stereo_depth
        or args.object_mask_annotation_3d
        or args.depth_model
        or args.stereo_model
        or args.annotation_model
    )
    enable_depth = args.depth_estimation or bool(args.depth_model) or not any_selector
    enable_stereo = args.stereo_depth or bool(args.stereo_model) or not any_selector
    enable_annotation = (
        args.object_mask_annotation_3d or bool(args.annotation_model) or not any_selector
    )

    if args.depth_model and enable_depth:
        depth_estimation.configure_model(args.depth_model)
    if args.stereo_model and enable_stereo:
        stereo_depth.configure_model(args.stereo_model)
    if args.annotation_model and enable_annotation:
        object_mask_annotation_3d.configure_model(args.annotation_model)

    _app = create_app(
        enable_depth_estimation=enable_depth,
        enable_stereo_depth=enable_stereo,
        enable_object_mask_annotation_3d=enable_annotation,
    )
    uvicorn.run(_app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
