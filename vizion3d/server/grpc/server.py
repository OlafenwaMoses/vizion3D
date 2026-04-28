import io
import logging
from concurrent import futures

import grpc
import numpy as np
from PIL import Image

from vizion3d.lifting import DepthEstimation, DepthEstimationCommand
from vizion3d.lifting.defaults import DEFAULT_DEPTH_MODEL_BACKEND
from vizion3d.lifting.utils import create_mesh_ply_binary, create_ply_binary
from vizion3d.proto import lifting_pb2, lifting_pb2_grpc


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


class LiftingServiceServicer(lifting_pb2_grpc.LiftingServiceServicer):
    def RunDepthEstimation(self, request, context):
        cmd = DepthEstimationCommand(
            image_input=request.image_bytes,
            model_backend=request.model_backend or DEFAULT_DEPTH_MODEL_BACKEND,
            return_depth_image=request.return_depth_image,
            return_point_cloud=request.return_point_cloud,
            return_mesh=request.return_mesh,
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


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    lifting_pb2_grpc.add_LiftingServiceServicer_to_server(LiftingServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("gRPC server running on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
