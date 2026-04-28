import numpy as np

from vizion3d.lifting.utils import create_mesh_ply_binary, create_ply_binary


def test_create_ply_binary_starts_with_header():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    ply = create_ply_binary(pts, cols)
    assert ply.startswith(b"ply\n")
    assert b"format binary_little_endian 1.0\n" in ply
    assert b"element vertex 2\n" in ply
    assert b"end_header\n" in ply


def test_create_ply_binary_byte_size():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    ply = create_ply_binary(pts, cols)
    header_end = ply.index(b"end_header\n") + len(b"end_header\n")
    binary_section = ply[header_end:]
    # Each vertex: 3 x float32 (12 bytes) + 3 x uint8 (3 bytes) = 15 bytes
    assert len(binary_section) == 2 * 15


def test_create_ply_binary_single_point():
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    cols = np.array([[128, 64, 32]], dtype=np.uint8)
    ply = create_ply_binary(pts, cols)
    assert b"element vertex 1\n" in ply
    header_end = ply.index(b"end_header\n") + len(b"end_header\n")
    assert len(ply[header_end:]) == 1 * 15


def test_create_mesh_ply_binary_header():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    cols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    ply = create_mesh_ply_binary(pts, cols, faces)
    assert ply.startswith(b"ply\n")
    assert b"element vertex 3\n" in ply
    assert b"element face 1\n" in ply
    assert b"property list uchar int vertex_indices\n" in ply
    assert b"end_header\n" in ply


def test_create_mesh_ply_binary_byte_size():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    cols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    ply = create_mesh_ply_binary(pts, cols, faces)
    header_end = ply.index(b"end_header\n") + len(b"end_header\n")
    binary_section = ply[header_end:]
    # 3 vertices * 15 bytes + 1 face * (1 uint8 count + 3 int32) = 45 + 13 = 58 bytes
    assert len(binary_section) == 3 * 15 + 1 * 13
