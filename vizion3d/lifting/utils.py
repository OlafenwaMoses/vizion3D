import io

import numpy as np


def create_ply_binary(points: np.ndarray, colors: np.ndarray) -> bytes:
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    buf = io.BytesIO()
    buf.write(header.encode("ascii"))

    data = np.zeros(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["r"] = colors[:, 0]
    data["g"] = colors[:, 1]
    data["b"] = colors[:, 2]

    buf.write(data.tobytes())
    return buf.getvalue()


def create_mesh_ply_binary(points: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> bytes:
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
    buf = io.BytesIO()
    buf.write(header.encode("ascii"))

    vertex_data = np.zeros(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["r"] = colors[:, 0]
    vertex_data["g"] = colors[:, 1]
    vertex_data["b"] = colors[:, 2]

    buf.write(vertex_data.tobytes())

    face_data = np.zeros(
        len(faces),
        dtype=[("count", "u1"), ("v0", "<i4"), ("v1", "<i4"), ("v2", "<i4")],
    )
    face_data["count"] = 3
    face_data["v0"] = faces[:, 0]
    face_data["v1"] = faces[:, 1]
    face_data["v2"] = faces[:, 2]

    buf.write(face_data.tobytes())
    return buf.getvalue()
