# Depth Estimation

**Category:** Lifting (2D → 3D)  
**Experimental:** No

Depth estimation predicts the per-pixel distance from the camera for every pixel in a 2D RGB image, producing a depth map and optionally unprojecting it into a 3D point cloud or surface mesh. vizion3d uses [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) as its default backend.

---

## Model backends

Default checkpoint download:
[depth_anything_v2_vitb.pth](https://github.com/OlafenwaMoses/vizion3D/releases/download/essentials-v1/depth_anything_v2_vitb.pth)

```bash
curl -L \
  https://github.com/OlafenwaMoses/vizion3D/releases/download/essentials-v1/depth_anything_v2_vitb.pth \
  -o depth_anything_v2_vitb.pth
```

| Value | What happens |
|---|---|
| *(default)* | Downloads the vizion3D release checkpoint (`depth_anything_v2_vitb.pth`) to `~/.cache/vizion3d/models/` on first use, then loads it directly |
| An HTTPS URL ending in `.pth` or `.pt` | Downloaded to the cache directory on first use, then loaded as a Depth Anything V2 checkpoint |
| A local `.pth` or `.pt` file path | Loaded directly as a Depth Anything V2 checkpoint — never downloaded |

Models are kept in memory after the first inference in the current process. Subsequent calls to any `DepthEstimation` instance reuse the loaded weights.

Set `VIZION3D_MODEL_CACHE` in your environment to change the default cache directory.

---

## Command parameters

`DepthEstimationCommand` is the input contract for this task.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_input` | `str \| bytes` | **Yes** | — | Image to process. Pass a file path string or raw image bytes. |
| `model_backend` | `str` | No | vizion3D release checkpoint URL | Model backend identifier. See [Model backends](#model-backends) above. |
| `return_depth_image` | `bool` | No | `False` | If `True`, the result includes a 16-bit grayscale Open3D Image of the depth map. |
| `return_point_cloud` | `bool` | No | `False` | If `True`, the result includes an Open3D PointCloud unprojected from the RGB-D image. |
| `return_mesh` | `bool` | No | `False` | If `True`, the result includes an Open3D TriangleMesh reconstructed from the point cloud via ball-pivoting. |
| `advanced_config` | `DepthEstimationAdvanceConfig` | No | PrimeSense defaults | Camera intrinsics and depth range settings. See [Advanced config](#10-advanced-config-camera-intrinsics-depth-range) below. |

---

## Result fields

`DepthEstimationResult` is the output contract for this task.

| Field | Type | Always present | Description |
|---|---|---|---|
| `depth_map` | `list[list[float]]` | Yes | Raw floating-point depth array, shape `[H][W]`. Values are relative (not metric) — closer objects have higher values for inverse-depth models. |
| `min_depth` | `float` | Yes | Minimum value in `depth_map`. |
| `max_depth` | `float` | Yes | Maximum value in `depth_map`. Guaranteed `max_depth >= min_depth`. |
| `backend_used` | `str` | Yes | Resolved model identifier that processed the request (local file path). |
| `depth_image` | `open3d.geometry.Image \| None` | When `return_depth_image=True` | 16-bit grayscale image, dtype `uint16`, shape `(H, W)`. The full 0–65535 range maps to `[min_depth, max_depth]`. |
| `point_cloud` | `open3d.geometry.PointCloud \| None` | When `return_point_cloud=True` | Coloured 3D point cloud unprojected from the RGB-D image using the intrinsics in `advanced_config`. Coordinates are in metres. |
| `mesh` | `open3d.geometry.TriangleMesh \| None` | When `return_mesh=True` | Triangle mesh surface reconstructed from the point cloud via ball-pivoting. Includes vertex colours. |
| `point_cloud_scale` | `float` | Yes | Scale factor: multiply any distance measured between two points in the point cloud by this value to get the equivalent distance in metres. Always `1.0` — Open3D produces point cloud coordinates directly in metres. |

---

## 1. Direct Python import — image bytes

The most common usage: read an image file into bytes and dispatch the command.

```python
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand

with open("scene.png", "rb") as f:
    img_bytes = f.read()

cmd = DepthEstimationCommand(image_input=img_bytes)
result = DepthEstimation().run(cmd)

print(f"Depth map shape : {len(result.depth_map)} rows × {len(result.depth_map[0])} cols")
print(f"Depth range     : {result.min_depth:.4f} → {result.max_depth:.4f}")
print(f"Backend used    : {result.backend_used}")
```

---

## 2. Direct Python import — file path

Pass a file path string instead of bytes; the handler opens it automatically.

```python
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand

cmd = DepthEstimationCommand(image_input="scene.png")
result = DepthEstimation().run(cmd)

print(f"Depth range: {result.min_depth:.4f} → {result.max_depth:.4f}")
```

---

## 3. Depth image (16-bit PNG)

Request a 16-bit grayscale Open3D Image of the depth map for visualization or downstream processing.

```python
import numpy as np
from PIL import Image as PILImage
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand

cmd = DepthEstimationCommand(
    image_input="scene.png",
    return_depth_image=True,
)
result = DepthEstimation().run(cmd)

# result.depth_image is an open3d.geometry.Image (uint16)
depth_array = np.asarray(result.depth_image)   # shape (H, W), dtype uint16
print(f"Depth image shape: {depth_array.shape}, dtype: {depth_array.dtype}")

# Save as 16-bit PNG via PIL
PILImage.fromarray(depth_array).save("depth.png")
```

---

## 4. Point cloud

Request a coloured 3D point cloud unprojected from the RGB-D image. Distances between points are in metres (`point_cloud_scale == 1.0`).

```python
import numpy as np
import open3d as o3d
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand

cmd = DepthEstimationCommand(
    image_input="scene.png",
    return_point_cloud=True,
)
result = DepthEstimation().run(cmd)

pcd = result.point_cloud                          # open3d.geometry.PointCloud
points = np.asarray(pcd.points)                   # shape (N, 3), float64, in metres
colors = np.asarray(pcd.colors)                   # shape (N, 3), float64, range [0, 1]

print(f"Points : {len(points)}")
print(f"Scale  : {result.point_cloud_scale} metre per unit")

# Measure real-world distance between two points
dist_metres = np.linalg.norm(points[0] - points[1]) * result.point_cloud_scale
print(f"Distance p0→p1: {dist_metres:.4f} m")

# Save as PLY
o3d.io.write_point_cloud("scene.ply", pcd)
```

---

## 5. Surface mesh

Request a triangulated mesh reconstructed from the point cloud via ball-pivoting. Includes vertex colours.

```python
import open3d as o3d
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand

cmd = DepthEstimationCommand(
    image_input="scene.png",
    return_mesh=True,
)
result = DepthEstimation().run(cmd)

mesh = result.mesh                                # open3d.geometry.TriangleMesh
print(f"Vertices  : {len(mesh.vertices)}")
print(f"Triangles : {len(mesh.triangles)}")

# Save as PLY
o3d.io.write_triangle_mesh("scene_mesh.ply", mesh)
```

---

## 6. All outputs at once

All three optional outputs can be requested in a single inference pass.

```python
import numpy as np
import open3d as o3d
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand

cmd = DepthEstimationCommand(
    image_input="scene.png",
    return_depth_image=True,
    return_point_cloud=True,
    return_mesh=True,
)
result = DepthEstimation().run(cmd)

# Depth map
print(f"Depth range : {result.min_depth:.4f} → {result.max_depth:.4f}")

# 16-bit depth image
depth_arr = np.asarray(result.depth_image)        # uint16 (H, W)

# Point cloud
pcd = result.point_cloud
o3d.io.write_point_cloud("scene.ply", pcd)

# Mesh
o3d.io.write_triangle_mesh("scene_mesh.ply", result.mesh)
```

---

## 7. Custom model backend

Use a local `.pth` checkpoint or a remote URL to a `.pth` file.

```python
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand

# Local checkpoint
cmd = DepthEstimationCommand(
    image_input="scene.png",
    model_backend="/models/depth_anything_v2_vitl.pth",
)
result = DepthEstimation().run(cmd)
print(f"Backend: {result.backend_used}")

# Remote checkpoint URL (downloaded and cached on first use)
cmd = DepthEstimationCommand(
    image_input="scene.png",
    model_backend=(
        "https://github.com/OlafenwaMoses/vizion3D/releases/download/"
        "essentials-v1/depth_anything_v2_vitb.pth"
    ),
)
result = DepthEstimation().run(cmd)
print(f"Backend: {result.backend_used}")
```

---

## 8. REST API

Start the server with all REST features enabled:

**pip / Poetry**
```bash
vizion3d-serve-rest
```

**uv**
```bash
uv run vizion3d-serve-rest
```

To preload a depth-estimation checkpoint into memory at startup, pass
`--depth_model`. This also enables the depth-estimation endpoint. If this flag
is omitted, the default vizion3D release model is downloaded on first inference
and cached under `~/.cache/vizion3d/models/`.

```bash
uv run vizion3d-serve-rest --depth_model /models/depth_anything_v2_vitb.pth
```

The REST server can also expose only selected features. If none of
`--depth_estimation`, `--stereo_depth`, `--depth_model`, or `--stereo_model` is
provided, all features are enabled. If any of those flags is provided, only the
selected features are enabled. A model path flag selects and preloads its
feature:

```bash
# Only POST /lifting/depth-estimation
uv run vizion3d-serve-rest --depth_estimation

# Only depth estimation, with the model loaded before the first request
uv run vizion3d-serve-rest \
  --depth_estimation \
  --depth_model /models/depth_anything_v2_vitb.pth
```

Send a request with `multipart/form-data`:

```bash
curl -X POST "http://localhost:8000/lifting/depth-estimation" \
  -F "image=@scene.png" \
  -F "return_point_cloud=true" \
  -F "return_mesh=true"
```

The response is a JSON-serialised `DepthEstimationResult`. Binary fields (`depth_image`, `point_cloud`, `mesh`) are base64-encoded in the JSON response.

---

## 9. gRPC API

Start the server:

**pip / Poetry**
```bash
vizion3d-serve-grpc
```

**uv**
```bash
uv run vizion3d-serve-grpc
```

Call from any gRPC client using the generated stubs:

```python
import grpc
from vizion3d.proto import lifting_pb2, lifting_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = lifting_pb2_grpc.LiftingServiceStub(channel)

with open("scene.png", "rb") as f:
    img_bytes = f.read()

request = lifting_pb2.DepthEstimationRequest(
    image_bytes=img_bytes,
    return_point_cloud=True,
    return_mesh=True,
)

response = stub.RunDepthEstimation(request)
print(f"Min depth : {response.min_depth}")
print(f"Max depth : {response.max_depth}")
print(f"Backend   : {response.backend_used}")
```

---

## 10. Advanced config: camera intrinsics & depth range

`DepthEstimationAdvanceConfig` lets you supply the actual camera intrinsics and depth range for your sensor, replacing the built-in PrimeSense defaults. This is required for accurate metric 3D geometry when your camera is not a 640×480 PrimeSense sensor.

```python
from vizion3d.lifting import (
    DepthEstimation,
    DepthEstimationAdvanceConfig,
    DepthEstimationCommand,
)

result = DepthEstimation().run(
    DepthEstimationCommand(
        image_input="scene.png",
        return_point_cloud=True,
        advanced_config=DepthEstimationAdvanceConfig(
            fx=909.15,
            fy=908.48,
            cx=640.0,
            cy=360.0,
            depth_trunc=6.0,
        ),
    )
)
```

The same config is available in the REST and gRPC entry points. See [Advanced Config](depth_estimation_advanced_config.md) for the full field reference, formulas, entry-point examples, and camera presets.

---

## Known limitations

- **Relative depth only** — the default monocular backend produces relative (inverse) depth, not metric depth. Point cloud distances are internally consistent but not calibrated to real-world scale without a known reference distance.
- **Ball-pivoting mesh quality** — the mesh reconstructor works best on dense, evenly sampled point clouds. Sparse or noisy clouds may produce gaps or missing faces.
- **Python 3.12 required for Open3D** — `return_depth_image`, `return_point_cloud`, and `return_mesh` require Open3D, which currently only supports Python 3.12 in this project.
