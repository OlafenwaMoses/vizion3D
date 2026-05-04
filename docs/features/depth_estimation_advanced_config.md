# Advanced Config: Camera Intrinsics & Depth Range

`DepthEstimationAdvanceConfig` lets you override the camera intrinsics and depth range parameters that control how a raw depth map is lifted into a 3D point cloud. Without it, vizion3d uses built-in PrimeSense defaults; with it, you can match your actual camera and scene requirements precisely.

---

## Background: the pinhole camera model

Every point in a point cloud is computed by inverting the pinhole camera projection. Given a pixel at image coordinates `(u, v)` with a depth value `d` (in metres), its 3D position `(X, Y, Z)` is:

```
Z = d
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
```

All four intrinsic parameters — `fx`, `fy`, `cx`, `cy` — appear in this formula. Getting them wrong produces a point cloud that is geometrically distorted: correct topology but wrong angles, skewed shapes, or objects that appear compressed or stretched.

---

## Config fields

### `fx` — horizontal focal length (pixels)

**Default:** `525.0`

The horizontal focal length of the camera in pixels. It is the product of the physical focal length (mm) and the horizontal pixel density (pixels/mm). A larger `fx` means the camera has a narrower horizontal field of view; the same scene width maps to fewer pixels.

**Effect on the point cloud:** `fx` controls the horizontal spread of 3D points. If `fx` is too small, the point cloud is horizontally compressed. If too large, it is horizontally stretched.

**How to find it:** Use your camera's calibration matrix `K[0][0]`, or compute it from the horizontal field of view `FoV_h`:

```
fx = (image_width / 2) / tan(FoV_h / 2)
```

---

### `fy` — vertical focal length (pixels)

**Default:** `525.0`

The vertical focal length in pixels. For cameras with square pixels, `fy ≈ fx`. Cameras with non-square sensors may have `fy ≠ fx`.

**Effect on the point cloud:** Controls vertical spread analogously to `fx`. Incorrect `fy` produces vertically compressed or stretched geometry.

**How to find it:** `K[1][1]` from the calibration matrix, or:

```
fy = (image_height / 2) / tan(FoV_v / 2)
```

---

### `cx` — horizontal principal point (pixels)

**Default:** `319.5`

The horizontal image coordinate of the optical axis — ideally the exact centre of the sensor. For a 640-wide image the ideal value is `319.5`; for a 1920-wide image it is typically near `959.5`.

**Effect on the point cloud:** Shifts the entire point cloud left or right. A wrong `cx` makes the scene appear to be viewed from an off-centre vantage point, introducing a lateral tilt.

---

### `cy` — vertical principal point (pixels)

**Default:** `239.5`

The vertical image coordinate of the optical axis. For a 480-tall image the ideal value is `239.5`.

**Effect on the point cloud:** Shifts the entire point cloud up or down. Like `cx`, an incorrect value introduces a tilt — vertical in this case.

---

### `depth_scale` — depth value scale factor

**Default:** `1000.0`

The divisor applied to the raw uint16 depth buffer before passing depth values to Open3D's `RGBDImage.create_from_color_and_depth`. Open3D divides the stored integer depth by `depth_scale` to obtain a value in metres. The default of `1000.0` means the uint16 range `[0, 65535]` maps to `[0, 65.535]` metres.

**Effect on the point cloud:** Changing `depth_scale` rescales all Z values (and therefore X/Y values, since `X = (u - cx) * Z / fx`). Doubling `depth_scale` halves all distances. This does **not** change the relative shape of the cloud — only the metric scale.

**When to adjust:** Only change this if you are supplying a depth buffer in a different unit (e.g. centimetres instead of millimetres). In vizion3d the depth map is internally normalised before being encoded into uint16, so the default `1000.0` is correct for the standard workflow.

---

### `depth_trunc` — maximum depth clip distance (metres)

**Default:** `10.0`

Points with a depth value greater than `depth_trunc` metres are discarded by Open3D before building the point cloud. This controls the far clipping plane.

**Effect on the point cloud:** Lowering `depth_trunc` removes distant background points and produces a denser, cleaner cloud for near objects. Setting it to a very small value will discard almost all points. Setting it too large can include noisy, low-confidence depth estimates at the scene boundary.

**Practical guidance:**
- Indoor close-up scenes: `2.0–5.0` m
- Room-scale scenes: `5.0–10.0` m (default)
- Outdoor or large-scale: `10.0–30.0` m

---

## Default values and PrimeSense

The built-in defaults match the **PrimeSense / Microsoft Kinect v1** sensor at 640×480 VGA resolution:

| Parameter | Default | PrimeSense VGA |
|---|---|---|
| `fx` | `525.0` | 525.0 px |
| `fy` | `525.0` | 525.0 px |
| `cx` | `319.5` | 319.5 px |
| `cy` | `239.5` | 239.5 px |
| `depth_scale` | `1000.0` | — |
| `depth_trunc` | `10.0` | — |

These are reasonable placeholders for any RGB camera with a ~60° horizontal FoV. For accurate metric reconstruction, always supply intrinsics from your actual camera calibration.

---

## Usage: Direct Python

```python
from vizion3d.lifting import (
    DepthEstimation,
    DepthEstimationAdvanceConfig,
    DepthEstimationCommand,
)

# Full custom intrinsics (e.g. Intel RealSense D435 at 1280×720)
config = DepthEstimationAdvanceConfig(
    fx=909.15,
    fy=908.48,
    cx=640.0,
    cy=360.0,
    depth_scale=1000.0,
    depth_trunc=6.0,
)

with open("scene.png", "rb") as f:
    img_bytes = f.read()

result = DepthEstimation().run(
    DepthEstimationCommand(
        image_input=img_bytes,
        return_point_cloud=True,
        advanced_config=config,
    )
)

import numpy as np
points = np.asarray(result.point_cloud.points)
print(f"Points: {len(points)}")
```

Partial overrides work too — unspecified fields keep their defaults:

```python
# Only change depth_trunc; everything else stays at PrimeSense defaults
result = DepthEstimation().run(
    DepthEstimationCommand(
        image_input=img_bytes,
        return_point_cloud=True,
        advanced_config=DepthEstimationAdvanceConfig(depth_trunc=3.0),
    )
)
```

---

## Usage: REST API

All six config parameters are optional form fields on the `POST /lifting/depth-estimation` endpoint.

```bash
# Full custom intrinsics
curl -X POST "http://localhost:8000/lifting/depth-estimation" \
  -F "image=@scene.png" \
  -F "return_point_cloud=true" \
  -F "fx=909.15" \
  -F "fy=908.48" \
  -F "cx=640.0" \
  -F "cy=360.0" \
  -F "depth_scale=1000.0" \
  -F "depth_trunc=6.0"
```

Partial overrides — omit any field to keep its default:

```bash
# Only override depth_trunc
curl -X POST "http://localhost:8000/lifting/depth-estimation" \
  -F "image=@scene.png" \
  -F "return_point_cloud=true" \
  -F "depth_trunc=3.0"
```

Python `requests` equivalent:

```python
import requests

with open("scene.png", "rb") as f:
    img_bytes = f.read()

response = requests.post(
    "http://localhost:8000/lifting/depth-estimation",
    files={"image": ("scene.png", img_bytes, "image/png")},
    data={
        "return_point_cloud": "true",
        "fx": "909.15",
        "fy": "908.48",
        "cx": "640.0",
        "cy": "360.0",
        "depth_trunc": "6.0",
    },
)
data = response.json()
print(f"Depth range: {data['min_depth']:.4f} → {data['max_depth']:.4f}")
```

---

## Usage: gRPC API

The `DepthEstimationAdvanceConfig` proto message mirrors the Python model. All fields are `optional`, so any omitted field falls back to the server-side default.

```python
import grpc
from vizion3d.proto import lifting_pb2, lifting_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = lifting_pb2_grpc.LiftingServiceStub(channel)

with open("scene.png", "rb") as f:
    img_bytes = f.read()

# Full custom intrinsics
request = lifting_pb2.DepthEstimationRequest(
    image_bytes=img_bytes,
    return_point_cloud=True,
    advanced_config=lifting_pb2.DepthEstimationAdvanceConfig(
        fx=909.15,
        fy=908.48,
        cx=640.0,
        cy=360.0,
        depth_scale=1000.0,
        depth_trunc=6.0,
    ),
)
response = stub.RunDepthEstimation(request)
print(f"Depth range: {response.min_depth:.4f} → {response.max_depth:.4f}")
```

Partial override — only `depth_trunc`:

```python
request = lifting_pb2.DepthEstimationRequest(
    image_bytes=img_bytes,
    return_point_cloud=True,
    advanced_config=lifting_pb2.DepthEstimationAdvanceConfig(depth_trunc=3.0),
)
response = stub.RunDepthEstimation(request)
```

---

## How to get your camera intrinsics

### Option 1: camera datasheet or SDK

Most camera SDKs expose the intrinsic matrix directly:

```python
# Intel RealSense
import pyrealsense2 as rs
pipeline = rs.pipeline()
profile  = pipeline.start()
intr     = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
config   = DepthEstimationAdvanceConfig(
    fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy
)
```

### Option 2: OpenCV calibration

Run a standard checkerboard calibration with `cv2.calibrateCamera`. The returned `camera_matrix` is:

```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

```python
import cv2
import numpy as np

# After calibrating…
_, camera_matrix, _, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

config = DepthEstimationAdvanceConfig(
    fx=float(camera_matrix[0, 0]),
    fy=float(camera_matrix[1, 1]),
    cx=float(camera_matrix[0, 2]),
    cy=float(camera_matrix[1, 2]),
)
```

### Option 3: approximate from field of view

If you know the camera's horizontal field of view `FoV_h` (in degrees) and image dimensions:

```python
import math

image_width  = 1920
image_height = 1080
fov_h_deg    = 69.0          # horizontal FoV in degrees

fx = (image_width  / 2) / math.tan(math.radians(fov_h_deg / 2))
fy = fx                      # assumes square pixels
cx = image_width  / 2 - 0.5
cy = image_height / 2 - 0.5

config = DepthEstimationAdvanceConfig(fx=fx, fy=fy, cx=cx, cy=cy)
```

---

## Common camera presets

These are approximate values for common cameras. Always prefer calibrated values over these presets.

| Camera | Resolution | fx | fy | cx | cy |
|---|---|---|---|---|---|
| PrimeSense / Kinect v1 | 640×480 | 525.0 | 525.0 | 319.5 | 239.5 |
| Intel RealSense D415 | 1920×1080 | 1382.0 | 1382.0 | 960.5 | 540.5 |
| Intel RealSense D435 | 1280×720 | 909.0 | 908.0 | 640.0 | 360.0 |
| iPhone 14 wide (approx.) | 4032×3024 | 5500.0 | 5500.0 | 2016.0 | 1512.0 |
| Webcam 1080p (typical) | 1920×1080 | 1400.0 | 1400.0 | 960.0 | 540.0 |
