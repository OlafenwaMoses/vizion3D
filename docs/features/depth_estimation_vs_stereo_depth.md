# Depth Estimation vs Stereo Depth

vizion3d offers two different approaches to recovering depth from images. This page explains how they differ, when to choose each one, and what the practical trade-offs are.

---

## What is Depth Estimation?

[Depth Estimation](depth_estimation.md) uses a single RGB image and a monocular neural network (Depth Anything V2) to predict a per-pixel depth map.  The network infers depth from visual cues — perspective, texture gradients, occlusion — without any geometric information about the camera.

The output is **relative** (inverse) depth: closer objects have higher values, but the actual distances in metres are unknown.  Point clouds built from monocular depth are geometrically consistent in shape but not anchored to a real-world scale.

## What is Stereo Depth?

[Stereo Depth](stereo_depth.md) takes a **rectified pair** of images (left and right, from two cameras with a known physical separation) and finds corresponding pixels across both views.  The horizontal shift between matched pixels — called **disparity** — directly encodes depth through the stereo geometry formula:

```
depth_m = baseline_mm × focal_length_px / disparity_px / 1000
```

Provided the camera calibration is accurate, the output is **real metric depth in metres** — every point in the point cloud has a physically meaningful distance from the camera.

---

## Side-by-side comparison

| | Stereo Depth (S2M2) | Depth Estimation (Depth Anything V2) |
|---|---|---|
| **Input** | Rectified left + right image pair | Single RGB image |
| **Depth type** | Metric (real metres) | Relative (inverse depth, arbitrary scale) |
| **Coordinate system** | Camera space | Camera space |
| **Units** | Metres (real) | Metres (fictitious — mapped to `[0, depth_trunc]`) |
| **Object at 2.4 m reads as 2.4 m** | Yes — if calibration is correct | No — depends on scene content |
| **Scale factor to world** | 1.0 (accurate) | Unknown, scene-dependent |
| **`point_cloud_scale` field** | 1.0 (accurate) | 1.0 (misleading — not real metres) |
| **Shape / topology correct** | Yes | Yes, if correct intrinsics supplied via `DepthEstimationAdvanceConfig` |
| **Camera calibration needed** | Yes — `focal_length`, `baseline`, `cx`, `cy` | Optional — only affects point cloud geometry |
| **Input requirements** | Stereo rig, rectified images | Any single photo |
| **Depth completeness** | Gaps in occluded / textureless regions | Dense — every pixel has a prediction |
| **Runtime** | Moderate (transformer-based matching) | Moderate (ViT-based encoder-decoder) |

---

## When to use each

### Use Depth Estimation when:
- You only have a single camera or single image.
- You need **dense** depth predictions (no holes from occlusion).
- You want to visualise relative 3D structure without exact scale.
- You are doing scene understanding, novel view synthesis, or artistic depth-of-field effects.

### Use Stereo Depth when:
- You have a stereo camera rig with known calibration.
- You need **real metric distances** — for robotics, measurement, AR anchoring.
- Textureless or low-contrast regions can be handled by the rig geometry.
- You need consistent scale across different scenes and camera positions.

---

## Working with point clouds from each

### Monocular point cloud (Depth Estimation)

```python
from vizion3d.lifting import DepthEstimation, DepthEstimationAdvanceConfig, DepthEstimationCommand
import numpy as np

result = DepthEstimation().run(
    DepthEstimationCommand(
        image_input="scene.png",
        return_point_cloud=True,
        advanced_config=DepthEstimationAdvanceConfig(
            fx=909.15, fy=908.48, cx=640.0, cy=360.0,
        ),
    )
)

points = np.asarray(result.point_cloud.points)  # shape (N, 3)
# point_cloud_scale == 1.0, but distances are NOT real metres —
# the depth model output is relative and mapped to depth_trunc.
print(f"point_cloud_scale: {result.point_cloud_scale}")  # 1.0 (misleading)
```

### Stereo point cloud (Stereo Depth)

```python
from vizion3d.stereo import StereoDepth, StereoDepthAdvancedConfig, StereoDepthCommand
import numpy as np

result = StereoDepth().run(
    StereoDepthCommand(
        left_image="left.png",
        right_image="right.png",
        return_point_cloud=True,
        advanced_config=StereoDepthAdvancedConfig(
            focal_length=1733.74,
            cx=792.27,
            cy=541.89,
            baseline=536.62,
        ),
    )
)

points = np.asarray(result.point_cloud.points)  # shape (N, 3), real metres
dist = np.linalg.norm(points[0] - points[1]) * result.point_cloud_scale
print(f"Real distance between p0 and p1: {dist:.4f} m")  # actual metres
print(f"point_cloud_scale: {result.point_cloud_scale}")   # 1.0 (accurate)
```

---

## Output differences at a glance

| Output field | Depth Estimation | Stereo Depth |
|---|---|---|
| `depth_map` | Relative depth (fictitious metres) | Metric depth (real metres) |
| `disparity_map` | Not present | Pixel disparity (always returned) |
| `min_depth` / `max_depth` | Relative range | Real range in metres |
| `point_cloud_scale` | 1.0 (misleading) | 1.0 (accurate) |
| `backend_used` | Local path to Depth Anything V2 `.pth` | Local path to S2M2 `.pth` |

---

## Advanced config comparison

Both tasks expose a camera configuration object, but the fields are different because the underlying geometry differs:

### DepthEstimationAdvanceConfig (monocular)

Controls how the relative depth map is converted into a point cloud:

```python
from vizion3d.lifting import DepthEstimationAdvanceConfig

cfg = DepthEstimationAdvanceConfig(
    fx=525.0,            # horizontal focal length (pixels)
    fy=525.0,            # vertical focal length (pixels)
    cx=319.5,            # principal point x
    cy=239.5,            # principal point y
    depth_scale=1000.0,  # uint16 → metres divisor
    depth_trunc=10.0,    # max depth in metres
)
```

### StereoDepthAdvancedConfig (stereo)

Controls the stereo geometry and point cloud quality filters:

```python
from vizion3d.stereo import StereoDepthAdvancedConfig

cfg = StereoDepthAdvancedConfig(
    focal_length=1000.0,  # focal length in pixels (fx = fy assumed)
    cx=640.0,             # principal point x
    cy=360.0,             # principal point y
    baseline=100.0,       # stereo baseline in millimetres
    doffs=0.0,            # disparity offset (Middlebury-style calibration)
    z_far=10.0,           # max depth in metres
    conf_threshold=0.1,   # min confidence score for point inclusion
    occ_threshold=0.5,    # min occlusion score for point inclusion
    scale_factor=1.0,     # input downscale for speed/quality tradeoff
)
```

The key stereo-only parameters are `baseline` (physical rig geometry) and `doffs` (calibration offset), which have no equivalent in monocular depth — they are meaningless without a second camera.
