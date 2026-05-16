# Lifting API Reference

The `lifting` module exposes tasks that convert 2D image data into 3D representations — depth maps, point clouds, and meshes.

All point clouds emitted by lifting tasks use OpenGL/viewer camera space: `X+` right, `Y+` up, and `Z-` forward into the scene.

---

## DepthEstimation

The primary entry point for the depth estimation task. Instantiate once and call `.run()` with a `DepthEstimationCommand`.

::: vizion3d.lifting.DepthEstimation

---

## DepthEstimationCommand

Input contract for the depth estimation task. All inference parameters are declared here.

::: vizion3d.lifting.commands.DepthEstimationCommand

---

## DepthEstimationAdvanceConfig

Camera intrinsics and depth range settings. Pass an instance of this model as `advanced_config` on `DepthEstimationCommand` to override the PrimeSense defaults used for point cloud unprojection. See [Camera Intrinsics Matrix](../concepts/camera_intrinsics.md) for a full explanation of `fx`, `fy`, `cx`, and `cy`.

::: vizion3d.lifting.models.DepthEstimationAdvanceConfig

---

## DepthEstimationResult

Output contract returned by `DepthEstimation.run()`. All fields are always present; optional geometry fields are `None` when the corresponding `return_*` flag was not set. Returned point clouds use OpenGL/viewer camera space: `X+` right, `Y+` up, `Z-` forward.

::: vizion3d.lifting.models.DepthEstimationResult
