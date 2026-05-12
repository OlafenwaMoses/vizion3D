"""
Shared serialisation helpers for the vizion3d REST server.

Converts Open3D geometry objects to wire-format bytes (PNG or binary PLY)
and base64-encodes arbitrary byte payloads for JSON transport.
"""

import base64
import io
import os
import tempfile

import numpy as np
from PIL import Image

from vizion3d.lifting.utils import create_ply_binary


def o3d_depth_image_to_png_bytes(o3d_image) -> bytes:
    """Encode an Open3D uint16 depth image as a PNG byte string."""
    arr = np.asarray(o3d_image)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def o3d_point_cloud_to_ply_bytes(pcd) -> bytes:
    """Serialise an Open3D PointCloud to binary PLY bytes."""
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return create_ply_binary(points, colors)


def b64(data: bytes | None) -> str | None:
    """Base64-encode *data*, or return ``None`` if *data* is ``None``."""
    return base64.b64encode(data).decode() if data is not None else None


def ply_bytes_to_o3d_point_cloud(ply_bytes: bytes):
    """Deserialise binary PLY bytes into an Open3D PointCloud.

    Writes to a temp file and uses ``open3d.io.read_point_cloud`` so that any
    valid PLY (not just vizion3d's own format) is accepted.

    Args:
        ply_bytes: Raw PLY file contents (binary or ASCII).

    Returns:
        ``open3d.geometry.PointCloud`` with points and (if present) colours.
    """
    import open3d as o3d

    fd, path = tempfile.mkstemp(suffix=".ply")
    try:
        os.write(fd, ply_bytes)
        os.close(fd)
        pcd = o3d.io.read_point_cloud(path)
    finally:
        os.unlink(path)
    return pcd


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Encode a boolean 2-D mask as an 8-bit grayscale PNG.

    ``True`` pixels become 255; ``False`` pixels become 0.

    Args:
        mask: Boolean array, shape ``(H, W)``.

    Returns:
        PNG-encoded bytes.
    """
    buf = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()
