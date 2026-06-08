"""
Shared geometry helpers for the annotation tasks.

These functions are segmentation-agnostic — they handle camera intrinsics
resolution, point-cloud array extraction, synthetic front-view rendering, and
back-projection of 3D points into image pixels.  Both
:class:`~vizion3d.annotation.handlers.ObjectMaskAnnotation3DHandler` (instance
segmentation) and
:class:`~vizion3d.annotation.scene_handlers.SceneMaskAnnotation3DHandler`
(semantic segmentation) build on them.

Intrinsics are duck-typed: any pydantic config exposing ``fx``, ``fy``, ``cx``,
``cy`` works.  The resolve/derive helpers return a copy of the *same* config
type via ``model_copy`` so task-specific fields are preserved.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _resolve_intrinsics(cfg, img_w: int, img_h: int):
    """Fill any None intrinsic fields from image dimensions (0.85×width FOV)."""
    fx = cfg.fx if cfg.fx is not None else img_w * 0.85
    fy = cfg.fy if cfg.fy is not None else img_w * 0.85
    cx = cfg.cx if cfg.cx is not None else img_w / 2.0
    cy = cfg.cy if cfg.cy is not None else img_h / 2.0
    return cfg.model_copy(update={"fx": fx, "fy": fy, "cx": cx, "cy": cy})


def _derive_intrinsics_from_cloud(pts: np.ndarray, cfg):
    """Derive any None intrinsics from the cloud's angular extent.

    Used when no image is provided — rendering requires concrete intrinsic
    values, so they must be computed from the cloud geometry first.  Derived
    values are consistent with the 0.85×width heuristic: for a cloud generated
    by DepthEstimation the recovered fx matches the auto-derived value within
    rounding.
    """
    if all(v is not None for v in (cfg.fx, cfg.fy, cfg.cx, cfg.cy)):
        return cfg

    if len(pts) == 0:
        return _resolve_intrinsics(cfg, 640, 480)

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = Z < 0
    if not valid.any():
        return _resolve_intrinsics(cfg, 640, 480)

    depth = -Z[valid]
    ax = X[valid] / depth
    ay = -Y[valid] / depth
    range_x = float(ax.max() - ax.min())

    _TARGET_W = 1024
    fx = (
        cfg.fx if cfg.fx is not None else (_TARGET_W / range_x if range_x > 0 else _TARGET_W * 0.85)
    )
    fy = cfg.fy if cfg.fy is not None else fx
    cx = cfg.cx if cfg.cx is not None else float(-ax.min() * fx)
    cy = cfg.cy if cfg.cy is not None else float(-ay.min() * fy)
    return cfg.model_copy(update={"fx": fx, "fy": fy, "cx": cx, "cy": cy})


def _extract_cloud_arrays(point_cloud) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(point_cloud.points)
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors).copy()
    else:
        colors = np.ones((len(pts), 3), dtype=np.float64)
    return pts, colors


def _render_front_view(pts: np.ndarray, colors: np.ndarray, cfg) -> Image.Image:
    """Synthesise a front-view RGB image by projecting XYZ+RGB into 2D.

    Canvas size includes the camera viewport implied by the principal point
    (roughly ``2*cx+1`` by ``2*cy+1``) and expands if projected points extend
    further.  Points are painted far-to-near so the nearest point wins at each
    pixel.  For DepthEstimation clouds (one point per pixel) the result is
    identical to the original photo when the original intrinsics are supplied.
    """
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = Z < 0
    if not valid.any():
        fallback_w = max(int(cfg.cx * 2) + 1, 1)
        fallback_h = max(int(cfg.cy * 2) + 1, 1)
        return Image.fromarray(np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8), mode="RGB")

    depth = -Z[valid]
    u = np.round(cfg.fx * X[valid] / depth + cfg.cx).astype(np.int32)
    v = np.round(cfg.cy - cfg.fy * Y[valid] / depth).astype(np.int32)

    viewport_w = int(np.ceil((cfg.cx or 0.0) * 2.0)) + 1
    viewport_h = int(np.ceil((cfg.cy or 0.0) * 2.0)) + 1
    img_w = max(viewport_w, int(u.max()) + 1, 1)
    img_h = max(viewport_h, int(v.max()) + 1, 1)
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v = u[in_bounds], v[in_bounds]
    z_vals = depth[in_bounds]
    rgb = (colors[valid][in_bounds] * 255).astype(np.uint8)

    order = np.argsort(z_vals)[::-1]
    canvas[v[order], u[order]] = rgb[order]

    return Image.fromarray(canvas, mode="RGB")


def _backproject(
    pts: np.ndarray,
    img_w: int,
    img_h: int,
    cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = Z < 0

    u_all = np.full(len(pts), -1, dtype=np.int32)
    v_all = np.full(len(pts), -1, dtype=np.int32)

    depth = -Z[valid]
    u_proj = np.round(cfg.fx * X[valid] / depth + cfg.cx).astype(np.int32)
    v_proj = np.round(cfg.cy - cfg.fy * Y[valid] / depth).astype(np.int32)

    u_all[valid] = u_proj
    v_all[valid] = v_proj

    in_bounds = valid & (u_all >= 0) & (u_all < img_w) & (v_all >= 0) & (v_all < img_h)

    in_bounds_idx = np.where(in_bounds)[0]
    return in_bounds_idx, u_all[in_bounds_idx], v_all[in_bounds_idx]


def _make_sub_cloud(pts: np.ndarray, colors: np.ndarray, indices: list[int], o3d):
    sub_pts = pts[indices]
    sub_cols = colors[indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub_pts)
    pcd.colors = o3d.utility.Vector3dVector(sub_cols)
    return pcd


def _make_full_cloud(pts: np.ndarray, colors: np.ndarray, o3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _clone_cloud(point_cloud, o3d):
    pts = np.asarray(point_cloud.points).copy()
    cols = (
        np.asarray(point_cloud.colors).copy()
        if point_cloud.has_colors()
        else np.ones((len(pts), 3))
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd
