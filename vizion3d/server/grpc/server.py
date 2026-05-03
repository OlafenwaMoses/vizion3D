"""
gRPC server for the vizion3d Lifting service.

Exposes two RPC methods:
- ``RunDepthEstimation`` — monocular depth from a single image.
- ``RunStereoDepth``     — metric depth from a rectified stereo image pair.

Start with::

    uv run vizion3d-serve-grpc
    # or
    python -m vizion3d.server.grpc.server
"""

import io
import logging
from concurrent import futures

import grpc
import numpy as np
from PIL import Image

from vizion3d.lifting import DepthEstimation, DepthEstimationCommand
from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_URL
from vizion3d.lifting.models import DepthEstimationAdvanceConfig
from vizion3d.lifting.utils import create_mesh_ply_binary, create_ply_binary
from vizion3d.proto import lifting_pb2, lifting_pb2_grpc
from vizion3d.stereo import StereoDepth, StereoDepthCommand
from vizion3d.stereo.defaults import DEFAULT_STEREO_MODEL_URL
from vizion3d.stereo.models import StereoDepthAdvancedConfig

# ── Shared serialisation helpers ──────────────────────────────────────────────


def _o3d_depth_image_to_png_bytes(o3d_image) -> bytes:
    """Encode an Open3D uint16 depth image as a PNG byte string."""
    arr = np.asarray(o3d_image)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _o3d_point_cloud_to_ply_bytes(pcd) -> bytes:
    """Serialise an Open3D PointCloud to binary PLY bytes."""
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return create_ply_binary(points, colors)


def _o3d_mesh_to_ply_bytes(mesh) -> bytes:
    """Serialise an Open3D TriangleMesh to binary PLY bytes."""
    points = np.asarray(mesh.vertices).astype(np.float32)
    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    return create_mesh_ply_binary(points, colors, faces)


# ── gRPC Servicer ─────────────────────────────────────────────────────────────


class LiftingServiceServicer(lifting_pb2_grpc.LiftingServiceServicer):
    """Implements the LiftingService proto RPC methods."""

    # ── RunDepthEstimation ────────────────────────────────────────────────────

    def RunDepthEstimation(self, request, context):
        """Handle a monocular depth estimation request.

        Unmarshals the proto config, dispatches through the CQRS command bus, and
        packs the result back into a proto response message.

        Args:
            request: ``DepthEstimationRequest`` proto message.
            context: gRPC server context.

        Returns:
            ``DepthEstimationResponse`` proto message.
        """
        base_cfg = DepthEstimationAdvanceConfig()
        if request.HasField("advanced_config"):
            proto_cfg = request.advanced_config
            base_cfg = DepthEstimationAdvanceConfig(
                fx=proto_cfg.fx if proto_cfg.HasField("fx") else base_cfg.fx,
                fy=proto_cfg.fy if proto_cfg.HasField("fy") else base_cfg.fy,
                cx=proto_cfg.cx if proto_cfg.HasField("cx") else base_cfg.cx,
                cy=proto_cfg.cy if proto_cfg.HasField("cy") else base_cfg.cy,
                depth_scale=(
                    proto_cfg.depth_scale
                    if proto_cfg.HasField("depth_scale")
                    else base_cfg.depth_scale
                ),
                depth_trunc=(
                    proto_cfg.depth_trunc
                    if proto_cfg.HasField("depth_trunc")
                    else base_cfg.depth_trunc
                ),
            )
        cmd = DepthEstimationCommand(
            image_input=request.image_bytes,
            model_backend=request.model_backend or DEFAULT_DEPTH_MODEL_URL,
            return_depth_image=request.return_depth_image,
            return_point_cloud=request.return_point_cloud,
            return_mesh=request.return_mesh,
            advanced_config=base_cfg,
        )
        result = DepthEstimation().run(cmd)

        response = lifting_pb2.DepthEstimationResponse(
            min_depth=result.min_depth,
            max_depth=result.max_depth,
            backend_used=result.backend_used,
        )
        for row in result.depth_map:
            response.depth_map.append(lifting_pb2.FloatRow(values=row))
        if result.depth_image is not None:
            response.depth_image = _o3d_depth_image_to_png_bytes(result.depth_image)
        if result.point_cloud is not None:
            response.point_cloud_ply = _o3d_point_cloud_to_ply_bytes(result.point_cloud)
        if result.mesh is not None:
            response.mesh_ply = _o3d_mesh_to_ply_bytes(result.mesh)
        return response

    # ── RunStereoDepth ────────────────────────────────────────────────────────

    def RunStereoDepth(self, request, context):
        """Handle a stereo depth estimation request.

        Unmarshals the proto config, dispatches through the CQRS command bus, and
        packs the result back into a proto response message.

        Args:
            request: ``StereoDepthRequest`` proto message.
            context: gRPC server context.

        Returns:
            ``StereoDepthResponse`` proto message.
        """
        base_cfg = StereoDepthAdvancedConfig()
        if request.HasField("advanced_config"):
            proto_cfg = request.advanced_config

            def _f(field: str, default):
                return getattr(proto_cfg, field) if proto_cfg.HasField(field) else default

            base_cfg = StereoDepthAdvancedConfig(
                focal_length=_f("focal_length", base_cfg.focal_length),
                cx=_f("cx", base_cfg.cx),
                cy=_f("cy", base_cfg.cy),
                baseline=_f("baseline", base_cfg.baseline),
                doffs=_f("doffs", base_cfg.doffs),
                z_far=_f("z_far", base_cfg.z_far),
                conf_threshold=_f("conf_threshold", base_cfg.conf_threshold),
                occ_threshold=_f("occ_threshold", base_cfg.occ_threshold),
                scale_factor=_f("scale_factor", base_cfg.scale_factor),
            )

        cmd = StereoDepthCommand(
            left_image=request.left_image_bytes,
            right_image=request.right_image_bytes,
            model_backend=request.model_backend or DEFAULT_STEREO_MODEL_URL,
            return_depth_image=request.return_depth_image,
            return_point_cloud=request.return_point_cloud,
            return_mesh=request.return_mesh,
            advanced_config=base_cfg,
        )
        result = StereoDepth().run(cmd)

        response = lifting_pb2.StereoDepthResponse(
            min_depth=result.min_depth,
            max_depth=result.max_depth,
            backend_used=result.backend_used,
        )
        for row in result.depth_map:
            response.depth_map.append(lifting_pb2.FloatRow(values=row))
        for row in result.disparity_map:
            response.disparity_map.append(lifting_pb2.FloatRow(values=row))
        if result.depth_image is not None:
            response.depth_image = _o3d_depth_image_to_png_bytes(result.depth_image)
        if result.point_cloud is not None:
            response.point_cloud_ply = _o3d_point_cloud_to_ply_bytes(result.point_cloud)
        if result.mesh is not None:
            response.mesh_ply = _o3d_mesh_to_ply_bytes(result.mesh)
        return response


# ── Server bootstrap ──────────────────────────────────────────────────────────

_MAX_MSG = 500 * 1024 * 1024  # 500 MB

_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MSG),
    ("grpc.max_receive_message_length", _MAX_MSG),
]


def serve():
    """Start the gRPC server on port 50051 and block until terminated."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=_GRPC_OPTIONS,
    )
    lifting_pb2_grpc.add_LiftingServiceServicer_to_server(LiftingServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("gRPC server running on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
