"""
gRPC server for the vizion3d Lifting service.

Exposes the LiftingService RPC methods:
- ``RunDepthEstimation``   — monocular depth from a single image.
- ``RunStereoDepth``       — metric depth from a rectified stereo image pair.
- ``RunObjectMaskAnnotation3D``— detect, instance-segment, and mask-annotate
                               objects in a point cloud (image optional).
- ``RunSceneMaskAnnotation3D`` — semantic-segment a scene and group point-cloud
                               points by class (image optional).
- ``RunScaleObservation``  — estimate metric scale from annotations.
- ``RunObject3DReconstruction`` — reconstruct one object as gray mesh and cloud.
- ``RunSceneComponents3DReconstruction`` — reconstruct detected scene objects.

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

from vizion3d.annotation import (
    ObjectMaskAnnotation3D,
    ObjectMaskAnnotation3DCommand,
    SceneMaskAnnotation3D,
    SceneMaskAnnotation3DCommand,
)
from vizion3d.annotation.defaults import DEFAULT_ANNOTATION_MODEL_URL
from vizion3d.annotation.models import ObjectMaskAnnotation3DConfig, SceneMaskAnnotation3DConfig
from vizion3d.annotation.scene_defaults import DEFAULT_SCENE_MODEL_URL
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand
from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_URL
from vizion3d.lifting.models import DepthEstimationAdvanceConfig
from vizion3d.lifting.utils import create_ply_binary
from vizion3d.observation import (
    ScaleObservation,
    ScaleObservationAdvancedConfig,
    ScaleObservationCommand,
)
from vizion3d.proto import lifting_pb2, lifting_pb2_grpc
from vizion3d.reconstruction import (
    Object3DReconstruction,
    Object3DReconstructionCommand,
    Object3DReconstructionConfig,
    SceneComponents3DReconstruction,
    SceneComponents3DReconstructionCommand,
    SceneComponents3DReconstructionConfig,
)
from vizion3d.server.rest.serialisation import trimesh_to_ply_bytes
from vizion3d.stereo import StereoDepth, StereoDepthCommand
from vizion3d.stereo.defaults import DEFAULT_STEREO_MODEL_URL
from vizion3d.stereo.models import StereoDepthAdvancedConfig

# ── Shared serialisation helpers ──────────────────────────────────────────────


def _o3d_depth_image_to_png_bytes(o3d_image) -> bytes:
    """Encode an Open3D uint16 depth image as a PNG byte string."""
    arr = np.asarray(o3d_image)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _o3d_point_cloud_to_ply_bytes(pcd) -> bytes:
    """Serialise an Open3D PointCloud to binary PLY bytes."""
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return create_ply_binary(points, colors)


def _mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Encode a boolean 2-D mask as an 8-bit grayscale PNG (255=object)."""
    buf = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255)).save(buf, format="PNG")
    return buf.getvalue()


def _mask_from_png_bytes(mask_bytes: bytes) -> np.ndarray:
    """Decode a PNG mask into a boolean 2-D array."""
    if not mask_bytes:
        return np.zeros((1, 1), dtype=bool)
    return np.asarray(Image.open(io.BytesIO(mask_bytes)).convert("L")) > 0


def _ply_bytes_to_o3d_point_cloud(ply_bytes: bytes):
    """Deserialise binary PLY bytes into an Open3D PointCloud via a temp file."""
    import os
    import tempfile

    import open3d as o3d

    fd, path = tempfile.mkstemp(suffix=".ply")
    try:
        os.write(fd, ply_bytes)
        os.close(fd)
        pcd = o3d.io.read_point_cloud(path)
    finally:
        os.unlink(path)
    return pcd


def _object_reconstruction_config(proto=None) -> Object3DReconstructionConfig:
    base = Object3DReconstructionConfig()
    if proto is None:
        return base

    def _f(field: str, default):
        return getattr(proto, field) if proto.HasField(field) else default

    return Object3DReconstructionConfig(
        max_input_dimension=_f(
            "max_input_dimension", base.max_input_dimension
        ),
        marching_cubes_resolution=_f(
            "marching_cubes_resolution", base.marching_cubes_resolution
        ),
        density_threshold=_f("density_threshold", base.density_threshold),
        point_count=_f("point_count", base.point_count),
        device=_f("device", base.device),
        foreground_ratio=_f("foreground_ratio", base.foreground_ratio),
        smoothing_iterations=_f(
            "smoothing_iterations", base.smoothing_iterations
        ),
        min_component_area_ratio=_f(
            "min_component_area_ratio", base.min_component_area_ratio
        ),
    )


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
                fx=proto_cfg.fx if proto_cfg.HasField("fx") else None,
                fy=proto_cfg.fy if proto_cfg.HasField("fy") else None,
                cx=proto_cfg.cx if proto_cfg.HasField("cx") else None,
                cy=proto_cfg.cy if proto_cfg.HasField("cy") else None,
            )
        cmd = DepthEstimationCommand(
            image_input=request.image_bytes,
            model_backend=request.model_backend or DEFAULT_DEPTH_MODEL_URL,
            return_depth_image=request.return_depth_image,
            return_point_cloud=request.return_point_cloud,
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
        return response

    # ── RunObjectAnnotation3D ─────────────────────────────────────────────────

    def RunObjectMaskAnnotation3D(self, request, context):
        """Handle a 3D object annotation request.

        Deserialises the input point cloud from PLY bytes, unmarshals the proto
        config, dispatches through the CQRS command bus, and packs the per-object
        annotation results back into a proto response.

        Args:
            request: ``ObjectAnnotation3DRequest`` proto message.
            context: gRPC server context.

        Returns:
            ``ObjectAnnotation3DResponse`` proto message.
        """
        point_cloud = _ply_bytes_to_o3d_point_cloud(request.point_cloud_ply)

        base_cfg = ObjectMaskAnnotation3DConfig()
        if request.HasField("advanced_config"):
            proto_cfg = request.advanced_config

            def _f(field: str, default):
                return getattr(proto_cfg, field) if proto_cfg.HasField(field) else default

            base_cfg = ObjectMaskAnnotation3DConfig(
                fx=_f("fx", base_cfg.fx),
                fy=_f("fy", base_cfg.fy),
                cx=_f("cx", base_cfg.cx),
                cy=_f("cy", base_cfg.cy),
                conf_threshold=_f("conf_threshold", base_cfg.conf_threshold),
                iou_threshold=_f("iou_threshold", base_cfg.iou_threshold),
            )

        # image_bytes empty in proto3 means no image was provided
        image_input = request.image_bytes if request.image_bytes else None

        cmd = ObjectMaskAnnotation3DCommand(
            point_cloud=point_cloud,
            image_input=image_input,
            model_backend=request.model_backend or DEFAULT_ANNOTATION_MODEL_URL,
            return_object_clouds=request.return_object_clouds,
            return_annotated_cloud=request.return_annotated_cloud,
            advanced_config=base_cfg,
        )
        result = ObjectMaskAnnotation3D().run(cmd)

        response = lifting_pb2.ObjectMaskAnnotation3DResponse(
            backend_used=result.backend_used,
        )

        for ann in result.annotations:
            item = lifting_pb2.MaskAnnotation3DItem(
                label=ann.label,
                class_id=ann.class_id,
                confidence=ann.confidence,
                bbox_2d=ann.bbox_2d,
                mask_image=_mask_to_png_bytes(ann.mask_2d),
                point_indices=ann.point_indices,
            )
            for coord in ann.point_coords:
                item.point_coords.append(lifting_pb2.FloatRow(values=coord))
            if ann.object_cloud is not None:
                item.object_cloud_ply = _o3d_point_cloud_to_ply_bytes(ann.object_cloud)
            response.annotations.append(item)

        if result.annotated_cloud is not None:
            response.annotated_cloud_ply = _o3d_point_cloud_to_ply_bytes(result.annotated_cloud)

        return response

    # ── RunSceneMaskAnnotation3D ──────────────────────────────────────────────

    def RunSceneMaskAnnotation3D(self, request, context):
        """Handle a 3D semantic scene annotation request.

        Deserialises the input point cloud from PLY bytes, unmarshals the proto
        config, dispatches through the CQRS command bus, and packs the per-class
        semantic annotation results back into a proto response.

        Args:
            request: ``SceneMaskAnnotation3DRequest`` proto message.
            context: gRPC server context.

        Returns:
            ``SceneMaskAnnotation3DResponse`` proto message.
        """
        point_cloud = _ply_bytes_to_o3d_point_cloud(request.point_cloud_ply)

        base_cfg = SceneMaskAnnotation3DConfig()
        if request.HasField("advanced_config"):
            proto_cfg = request.advanced_config

            def _f(field: str, default):
                return getattr(proto_cfg, field) if proto_cfg.HasField(field) else default

            base_cfg = SceneMaskAnnotation3DConfig(
                fx=_f("fx", base_cfg.fx),
                fy=_f("fy", base_cfg.fy),
                cx=_f("cx", base_cfg.cx),
                cy=_f("cy", base_cfg.cy),
                inference_size=_f("inference_size", base_cfg.inference_size),
            )

        image_input = request.image_bytes if request.image_bytes else None

        cmd = SceneMaskAnnotation3DCommand(
            point_cloud=point_cloud,
            image_input=image_input,
            model_backend=request.model_backend or DEFAULT_SCENE_MODEL_URL,
            return_region_clouds=request.return_region_clouds,
            return_annotated_cloud=request.return_annotated_cloud,
            advanced_config=base_cfg,
        )
        result = SceneMaskAnnotation3D().run(cmd)

        response = lifting_pb2.SceneMaskAnnotation3DResponse(
            backend_used=result.backend_used,
        )

        for ann in result.annotations:
            item = lifting_pb2.SemanticMaskAnnotation3DItem(
                label=ann.label,
                class_id=ann.class_id,
                bbox_2d=ann.bbox_2d,
                mask_image=_mask_to_png_bytes(ann.mask_2d),
                pixel_count=ann.pixel_count,
                point_indices=ann.point_indices,
            )
            for coord in ann.point_coords:
                item.point_coords.append(lifting_pb2.FloatRow(values=coord))
            if ann.region_cloud is not None:
                item.region_cloud_ply = _o3d_point_cloud_to_ply_bytes(ann.region_cloud)
            response.annotations.append(item)

        if result.annotated_cloud is not None:
            response.annotated_cloud_ply = _o3d_point_cloud_to_ply_bytes(result.annotated_cloud)

        return response

    # ── RunScaleObservation ─────────────────────────────────────────────────

    def RunScaleObservation(self, request, context):
        """Handle a ScaleObservation request."""
        from vizion3d.annotation.models import MaskAnnotation3D

        point_cloud = _ply_bytes_to_o3d_point_cloud(request.point_cloud_ply)
        annotations = []
        for item in request.annotations:
            annotations.append(
                MaskAnnotation3D(
                    label=item.label,
                    class_id=item.class_id,
                    confidence=item.confidence,
                    bbox_2d=list(item.bbox_2d),
                    mask_2d=_mask_from_png_bytes(item.mask_image),
                    point_indices=[],
                    point_coords=[list(row.values) for row in item.point_coords],
                )
            )

        def _field(name: str):
            return getattr(request, name) if request.HasField(name) else None

        cmd = ScaleObservationCommand(
            point_cloud=point_cloud,
            annotations=annotations,
            return_scaled_point_cloud=request.return_scaled_point_cloud,
            return_scaled_depth=request.return_scaled_depth,
            return_report=request.return_report,
            advanced_config=ScaleObservationAdvancedConfig(
                image_width=_field("image_width"),
                image_height=_field("image_height"),
                fx=_field("fx"),
                fy=_field("fy"),
                cx=_field("cx"),
                cy=_field("cy"),
            ),
        )
        result = ScaleObservation().run(cmd)
        response = lifting_pb2.ScaleObservationResponse(
            scale_factor=result.scale_factor,
            scale_confidence=result.scale_confidence,
            scale_confidence_reason=result.scale_confidence_reason,
            algorithm_version=result.algorithm_version,
            accepted_candidates=result.accepted_candidates,
            rejected_candidates=result.rejected_candidates,
        )
        for candidate in result.candidates:
            response.candidates.append(
                lifting_pb2.ScaleCandidateItem(
                    label=candidate.label,
                    dimension=candidate.dimension,
                    observed_relative=candidate.observed_relative,
                    prior_m=candidate.prior_m,
                    scale=candidate.scale,
                    weight=candidate.weight,
                    accepted=candidate.accepted,
                    rejection_reason=candidate.rejection_reason or "",
                )
            )
        if result.scaled_point_cloud is not None:
            response.scaled_point_cloud_ply = _o3d_point_cloud_to_ply_bytes(
                result.scaled_point_cloud
            )
        if result.scaled_depth_image is not None:
            response.scaled_depth_png = _o3d_depth_image_to_png_bytes(result.scaled_depth_image)
        return response

    def RunObject3DReconstruction(self, request, context):
        """Reconstruct a close-range object image as gray mesh and point cloud."""
        config = _object_reconstruction_config(
            request.advanced_config
            if request.HasField("advanced_config")
            else None
        )
        result = Object3DReconstruction().run(
            Object3DReconstructionCommand(
                image_input=request.image_bytes,
                model_bundle=request.model_bundle or None,
                advanced_config=config,
            )
        )
        return lifting_pb2.Object3DReconstructionResponse(
            mesh_ply=trimesh_to_ply_bytes(result.mesh),
            point_cloud_ply=_o3d_point_cloud_to_ply_bytes(result.point_cloud),
            vertex_count=result.vertex_count,
            face_count=result.face_count,
            point_count=result.point_count,
            backend_used=result.backend_used,
        )

    def RunSceneComponents3DReconstruction(self, request, context):
        """Reconstruct detected objects from one scene image."""
        base = SceneComponents3DReconstructionConfig()
        if request.HasField("advanced_config"):
            proto = request.advanced_config

            def _f(field: str, default):
                return getattr(proto, field) if proto.HasField(field) else default

            base = SceneComponents3DReconstructionConfig(
                max_input_dimension=_f(
                    "max_input_dimension", base.max_input_dimension
                ),
                max_objects=_f("max_objects", base.max_objects),
                confidence_threshold=_f(
                    "confidence_threshold", base.confidence_threshold
                ),
                padding_ratio=_f("padding_ratio", base.padding_ratio),
                object_config=_object_reconstruction_config(
                    proto.object_config
                    if proto.HasField("object_config")
                    else None
                ),
            )
        result = SceneComponents3DReconstruction().run(
            SceneComponents3DReconstructionCommand(
                image_input=request.image_bytes,
                model_bundle=request.model_bundle or None,
                depth_model_backend=request.depth_model_backend or None,
                annotation_model_backend=request.annotation_model_backend or None,
                advanced_config=base,
            )
        )
        response = lifting_pb2.SceneComponents3DReconstructionResponse(
            source_image_size=result.source_image_size,
            analysis_image_size=result.analysis_image_size,
            depth_backend_used=result.depth_backend_used,
            annotation_backend_used=result.annotation_backend_used,
            reconstruction_backend_used=result.reconstruction_backend_used,
        )
        for component in result.components:
            response.components.append(
                lifting_pb2.SceneComponent3DItem(
                    label=component.label,
                    class_id=component.class_id,
                    confidence=component.confidence,
                    bbox_2d=component.bbox_2d,
                    mesh_ply=trimesh_to_ply_bytes(component.mesh),
                    point_cloud_ply=_o3d_point_cloud_to_ply_bytes(
                        component.point_cloud
                    ),
                    vertex_count=component.vertex_count,
                    face_count=component.face_count,
                    point_count=component.point_count,
                )
            )
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
