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
