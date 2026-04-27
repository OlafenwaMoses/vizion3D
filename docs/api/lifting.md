# Lifting API Reference

The `lifting` module exposes tasks that convert 2D image data into 3D representations — depth maps, point clouds, and meshes.

---

## DepthEstimation

The primary entry point for the depth estimation task. Instantiate once and call `.run()` with a `DepthEstimationCommand`.

::: vision3d.lifting.DepthEstimation

---

## DepthEstimationCommand

Input contract for the depth estimation task. All inference parameters are declared here.

::: vision3d.lifting.commands.DepthEstimationCommand

---

## DepthEstimationResult

Output contract returned by `DepthEstimation.run()`. All fields are always present; optional geometry fields are `None` when the corresponding `return_*` flag was not set.

::: vision3d.lifting.models.DepthEstimationResult
