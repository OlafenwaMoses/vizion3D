# Stereo Depth

**Category:** Lifting (2D → 3D)  
**Experimental:** No

Stereo depth estimation recovers per-pixel **metric depth** (in metres) from a pair of rectified left/right RGB images by matching corresponding pixels across the two views and applying the stereo geometry formula:

```
depth_m = baseline_mm × focal_length_px / disparity_px / 1000
```

vizion3d uses [S2M2](https://github.com/Dongyeop-Yoo/S2M2) (Stereo Matching Model with Multi-scale transformer) as its stereo backend.  Unlike [Depth Estimation](depth_estimation.md), stereo depth produces **real-world metric distances** — provided the camera calibration parameters are correct.

---

## Model backends

| Value | What happens |
|---|---|
| *(default)* | Downloads the vizion3D release checkpoint (`stereo-depth-s2m2-L.pth`, the L variant) to `~/.cache/vizion3d/models/` on first use, then loads it |
| An HTTPS URL ending in `.pth` or `.pt` | Downloaded to the cache directory on first use, then loaded as an S2M2 checkpoint |
| A local `.pth` or `.pt` file path | Loaded directly — no download |

Models are kept in memory after the first inference.  Set `VIZION3D_MODEL_CACHE` to override the cache directory.

### S2M2 variants

The S2M2 architecture comes in four size variants.  The correct one is detected automatically from the checkpoint filename:

| Variant | Channels | Transformers | Speed | Quality |
|---|---|---|---|---|
| S (`-S.pth`) | 128 | 1 | Fastest | Good |
| M (`-M.pth`) | 192 | 2 | Fast | Better |
| L (`-L.pth`) | 256 | 3 | Balanced | Best (default) |
| XL (`-XL.pth`) | 384 | 3 | Slowest | Best |

---

## Command parameters

`StereoDepthCommand` is the input contract for this task.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `left_image` | `str \| bytes` | **Yes** | — | Left-camera image. Pass a file path string or raw image bytes. |
| `right_image` | `str \| bytes` | **Yes** | — | Right-camera image (same resolution, horizontally offset from `left_image`). |
| `model_backend` | `str` | No | vizion3D release checkpoint URL | S2M2 checkpoint. See [Model backends](#model-backends) above. |
| `return_depth_image` | `bool` | No | `False` | If `True`, the result includes a 16-bit grayscale Open3D Image of the depth map. |
| `return_point_cloud` | `bool` | No | `False` | If `True`, the result includes an Open3D PointCloud in metres. |
| `return_mesh` | `bool` | No | `False` | If `True`, the result includes an Open3D TriangleMesh reconstructed via ball-pivoting. |
| `advanced_config` | `StereoDepthAdvancedConfig` | No | 1280×720 @ 100 mm baseline defaults | Camera intrinsics and inference settings. See [Advanced config](#advanced-config) below. |

---

## Result fields

`StereoDepthResult` is the output contract.

| Field | Type | Always present | Description |
|---|---|---|---|
| `depth_map` | `list[list[float]]` | Yes | Metric depth in **metres**, shape `[H][W]`. Real-world distances (assuming correct calibration). |
| `disparity_map` | `list[list[float]]` | Yes | Raw disparity in **pixels**, shape `[H][W]`. Horizontal pixel offset between matched features. |
| `min_depth` | `float` | Yes | Minimum value in `depth_map` (metres). |
| `max_depth` | `float` | Yes | Maximum value in `depth_map`. Guaranteed `max_depth >= min_depth`. |
| `backend_used` | `str` | Yes | Resolved local file path of the checkpoint used. |
| `depth_image` | `open3d.geometry.Image \| None` | When `return_depth_image=True` | 16-bit grayscale image, dtype `uint16`. The full 0–65535 range maps to `[min_depth, max_depth]` in metres. |
| `point_cloud` | `open3d.geometry.PointCloud \| None` | When `return_point_cloud=True` | Coloured 3D point cloud, coordinates in **metres**. |
| `mesh` | `open3d.geometry.TriangleMesh \| None` | When `return_mesh=True` | Surface mesh from ball-pivoting. Includes vertex colours. |
| `point_cloud_scale` | `float` | Yes | Always `1.0` — stereo depth produces real metric coordinates. |

---

## 1. Direct Python import — image bytes

```python
from vizion3d.stereo import StereoDepth, StereoDepthCommand

with open("left.png", "rb") as f:
    left_bytes = f.read()
with open("right.png", "rb") as f:
    right_bytes = f.read()

cmd = StereoDepthCommand(left_image=left_bytes, right_image=right_bytes)
result = StereoDepth().run(cmd)

print(f"Depth range : {result.min_depth:.2f} → {result.max_depth:.2f} m")
print(f"Backend     : {result.backend_used}")
```

---

## 2. Direct Python import — file paths

```python
from vizion3d.stereo import StereoDepth, StereoDepthCommand

cmd = StereoDepthCommand(
    left_image="left.png",
    right_image="right.png",
)
result = StereoDepth().run(cmd)

print(f"Depth range: {result.min_depth:.2f} → {result.max_depth:.2f} m")
```

---

## 3. Disparity map

The raw disparity map (in pixels) is always returned alongside the depth map.

```python
import numpy as np
from vizion3d.stereo import StereoDepth, StereoDepthCommand

cmd = StereoDepthCommand(left_image="left.png", right_image="right.png")
result = StereoDepth().run(cmd)

disp = np.array(result.disparity_map)
print(f"Disparity range: {disp.min():.1f} → {disp.max():.1f} px")
```

---

## 4. Depth image (16-bit PNG)

```python
import numpy as np
from PIL import Image as PILImage
from vizion3d.stereo import StereoDepth, StereoDepthCommand

cmd = StereoDepthCommand(
    left_image="left.png",
    right_image="right.png",
    return_depth_image=True,
)
result = StereoDepth().run(cmd)

depth_array = np.asarray(result.depth_image)   # shape (H, W), dtype uint16
PILImage.fromarray(depth_array).save("depth.png")
```

---

## 5. Point cloud

Point coordinates are in **real metres** — `point_cloud_scale` is always `1.0`.

```python
import numpy as np
import open3d as o3d
from vizion3d.stereo import StereoDepth, StereoDepthAdvancedConfig, StereoDepthCommand

cmd = StereoDepthCommand(
    left_image="left.png",
    right_image="right.png",
    return_point_cloud=True,
    advanced_config=StereoDepthAdvancedConfig(
        focal_length=1733.74,
        cx=792.27,
        cy=541.89,
        baseline=536.62,   # mm
    ),
)
result = StereoDepth().run(cmd)

pcd = result.point_cloud
points = np.asarray(pcd.points)               # shape (N, 3), metres
print(f"Points: {len(points):,}")
print(f"Scale : {result.point_cloud_scale} m/unit")  # always 1.0

# Real-world distance between two points
dist = np.linalg.norm(points[0] - points[1]) * result.point_cloud_scale
print(f"p0→p1: {dist:.4f} m")

o3d.io.write_point_cloud("scene.ply", pcd)
```

---

## 6. Surface mesh

```python
import open3d as o3d
from vizion3d.stereo import StereoDepth, StereoDepthCommand

cmd = StereoDepthCommand(
    left_image="left.png",
    right_image="right.png",
    return_mesh=True,
)
result = StereoDepth().run(cmd)

mesh = result.mesh
print(f"Vertices  : {len(mesh.vertices)}")
print(f"Triangles : {len(mesh.triangles)}")
o3d.io.write_triangle_mesh("scene_mesh.ply", mesh)
```

---

## 7. All outputs at once

```python
import numpy as np
import open3d as o3d
from vizion3d.stereo import StereoDepth, StereoDepthCommand

cmd = StereoDepthCommand(
    left_image="left.png",
    right_image="right.png",
    return_depth_image=True,
    return_point_cloud=True,
    return_mesh=True,
)
result = StereoDepth().run(cmd)

print(f"Depth range : {result.min_depth:.2f} → {result.max_depth:.2f} m")
depth_arr = np.asarray(result.depth_image)    # uint16 (H, W)
o3d.io.write_point_cloud("scene.ply", result.point_cloud)
o3d.io.write_triangle_mesh("scene_mesh.ply", result.mesh)
```

---

## 8. Speed vs quality: scale factor

Use `scale_factor < 1.0` to downsample input before inference for faster results:

```python
from vizion3d.stereo import StereoDepth, StereoDepthAdvancedConfig, StereoDepthCommand

cmd = StereoDepthCommand(
    left_image="left.png",
    right_image="right.png",
    advanced_config=StereoDepthAdvancedConfig(
        scale_factor=0.5,   # half-resolution → ~3–4× faster
    ),
)
result = StereoDepth().run(cmd)
```

---

## 9. REST API

Start the server with all REST features enabled:

```bash
uv run vizion3d-serve-rest
```

To preload the stereo checkpoint into memory at startup, pass `--stereo_model`.
This also enables the stereo-depth endpoint. If this flag is omitted, the
default vizion3D release model is downloaded on first inference and cached under
`~/.cache/vizion3d/models/`.

```bash
uv run vizion3d-serve-rest \
  --stereo_model /models/stereo-depth-s2m2-L.pth
```

The REST server can expose only selected features. If none of
`--depth_estimation`, `--stereo_depth`, `--depth_model`, or `--stereo_model` is
provided, all features are enabled. If any of those flags is provided, only the
selected features are enabled. A model path flag selects and preloads its
feature:

```bash
# Only POST /lifting/stereo-depth
uv run vizion3d-serve-rest --stereo_depth

# Only stereo depth, with the model loaded before the first request
uv run vizion3d-serve-rest \
  --stereo_depth \
  --stereo_model /models/stereo-depth-s2m2-L.pth

# Enable both depth estimation and stereo depth explicitly
uv run vizion3d-serve-rest \
  --depth_estimation \
  --stereo_depth \
  --depth_model /models/depth_anything_v2_vitb.pth \
  --stereo_model /models/stereo-depth-s2m2-L.pth
```

Send a request with two image files:

```bash
curl -X POST "http://localhost:8000/lifting/stereo-depth" \
  -F "left_image=@left.png" \
  -F "right_image=@right.png" \
  -F "focal_length=1733.74" \
  -F "baseline=536.62" \
  -F "cx=792.27" \
  -F "cy=541.89" \
  -F "return_point_cloud=true"
```

The response is a JSON-serialised `StereoDepthResult`.  Binary fields (`depth_image`, `point_cloud_ply`, `mesh_ply`) are base64-encoded.

---

## 10. gRPC API

Start the server:

```bash
uv run vizion3d-serve-grpc
```

Call from a gRPC client:

```python
import grpc
from vizion3d.proto import lifting_pb2, lifting_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = lifting_pb2_grpc.LiftingServiceStub(channel)

with open("left.png", "rb") as f:
    left_bytes = f.read()
with open("right.png", "rb") as f:
    right_bytes = f.read()

request = lifting_pb2.StereoDepthRequest(
    left_image_bytes=left_bytes,
    right_image_bytes=right_bytes,
    return_point_cloud=True,
    advanced_config=lifting_pb2.StereoDepthAdvancedConfig(
        focal_length=1733.74,
        baseline=536.62,
        cx=792.27,
        cy=541.89,
    ),
)
response = stub.RunStereoDepth(request)
print(f"Min depth : {response.min_depth:.2f} m")
print(f"Max depth : {response.max_depth:.2f} m")
print(f"Backend   : {response.backend_used}")
```

---

## Advanced config

`StereoDepthAdvancedConfig` supplies the camera calibration needed for accurate metric depth.

| Field | Type | Default | Description |
|---|---|---|---|
| `focal_length` | `float` | `1000.0` | Focal length in pixels (assumes fx = fy). Override with your calibration. |
| `cx` | `float` | `640.0` | Principal point x (pixel column of optical axis). |
| `cy` | `float` | `360.0` | Principal point y (pixel row of optical axis). |
| `baseline` | `float` | `100.0` | Stereo baseline in **millimetres**. |
| `doffs` | `float` | `0.0` | Disparity offset (non-zero for Middlebury-style calibration). |
| `z_far` | `float` | `10.0` | Max depth in metres for point cloud. |
| `conf_threshold` | `float` | `0.1` | Min per-pixel confidence score for point cloud inclusion. |
| `occ_threshold` | `float` | `0.5` | Min occlusion score for point cloud inclusion. |
| `scale_factor` | `float` | `1.0` | Input downscale factor (`0.5` = half-res, ~3–4× faster). |

### How to obtain camera intrinsics

**From a calibration file (e.g. Middlebury):**
```python
# calib.txt format: cam0=[fx 0 cx; 0 fy cy; 0 0 1]
# baseline=B (mm), doffs=d
from vizion3d.stereo import StereoDepthAdvancedConfig

cfg = StereoDepthAdvancedConfig(
    focal_length=1733.74,   # from calib.txt
    cx=792.27,
    cy=541.89,
    baseline=536.62,        # B in mm
    doffs=0.0,              # d from calib.txt
)
```

**From Intel RealSense SDK:**
```python
import pyrealsense2 as rs

pipeline = rs.pipeline()
profile = pipeline.start()
left_stream = profile.get_stream(rs.stream.infrared, 1)
intrinsics = left_stream.as_video_stream_profile().get_intrinsics()

cfg = StereoDepthAdvancedConfig(
    focal_length=intrinsics.fx,
    cx=intrinsics.ppx,
    cy=intrinsics.ppy,
    baseline=50.0,   # RealSense D435 baseline ≈ 50 mm
)
```

**Approximation from field of view:**
```python
import math

hfov_deg = 90.0  # horizontal FOV from camera spec
image_width = 1280
focal_length = image_width / (2 * math.tan(math.radians(hfov_deg / 2)))

cfg = StereoDepthAdvancedConfig(
    focal_length=focal_length,
    cx=image_width / 2 - 0.5,
    cy=720 / 2 - 0.5,
    baseline=100.0,
)
```

---

## Known limitations

- **Rectified pairs required** — images must be stereo-rectified so corresponding points lie on the same horizontal scanline.  Un-rectified pairs will produce incorrect results.
- **Metric scale depends on calibration** — an incorrect `baseline` or `focal_length` scales all depth values uniformly.  Always use calibrated values for real applications.
- **Ball-pivoting mesh quality** — works best on dense, evenly sampled point clouds.  Sparse or noisy clouds from occluded regions may produce gaps or missing faces.
- **Python 3.12 required for Open3D** — `return_depth_image`, `return_point_cloud`, and `return_mesh` require Open3D, which currently only supports Python 3.12 in this project.
