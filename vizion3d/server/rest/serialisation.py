"""
Shared serialisation helpers for the vizion3d REST server.

Converts Open3D geometry objects to wire-format bytes (PNG or binary PLY)
and base64-encodes arbitrary byte payloads for JSON transport.
"""

import base64
import io

import numpy as np
from PIL import Image

from vizion3d.lifting.utils import create_mesh_ply_binary, create_ply_binary


def o3d_depth_image_to_png_bytes(o3d_image) -> bytes:
    """Encode an Open3D uint16 depth image as a PNG byte string."""
    arr = np.asarray(o3d_image)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def o3d_point_cloud_to_ply_bytes(pcd) -> bytes:
    """Serialise an Open3D PointCloud to binary PLY bytes."""
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return create_ply_binary(points, colors)


def o3d_mesh_to_ply_bytes(mesh) -> bytes:
    """Serialise an Open3D TriangleMesh to binary PLY bytes."""
    points = np.asarray(mesh.vertices).astype(np.float32)
    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    return create_mesh_ply_binary(points, colors, faces)


def b64(data: bytes | None) -> str | None:
    """Base64-encode *data*, or return ``None`` if *data* is ``None``."""
    return base64.b64encode(data).decode() if data is not None else None
