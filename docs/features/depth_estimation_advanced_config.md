# Advanced Config: Camera Intrinsics

`DepthEstimationAdvanceConfig` lets you supply the camera intrinsics that control how a raw depth map is lifted into a 3D point cloud. All four fields default to `None`, which causes the handler to auto-derive reasonable values from the input image dimensions. For accurate metric geometry, supply intrinsics from your actual camera calibration.

---

## Background: the pinhole camera model

> Not sure what `fx`, `fy`, `cx`, `cy` are? See the [Camera Intrinsics Matrix](../concepts/camera_intrinsics.md) reference for a full explanation of the K matrix and how to read it for your camera.

Every point in a point cloud is computed by inverting the pinhole camera projection. vizion3d emits OpenGL/viewer camera coordinates: `X+` right, `Y+` up, and `Z-` forward into the scene. Given a pixel at image coordinates `(u, v)` with a positive depth value `d` (in metres), its 3D position `(X, Y, Z)` is:

```
X = (u - cx) * d / fx
Y = (cy - v) * d / fy
Z = -d
```

All four intrinsic parameters — `fx`, `fy`, `cx`, `cy` — appear in this formula. Values that do not match your camera produce a point cloud that is geometrically distorted: correct topology but skewed angles, compressed shapes, or stretched geometry.

---

## Config fields

### `fx` — horizontal focal length (pixels)

**Default:** `None` (auto-derived as `image.width × 0.85`, ~63° horizontal FOV)

The horizontal focal length of the camera in pixels. A larger `fx` means the camera has a narrower horizontal field of view; the same scene width maps to fewer pixels.

**Effect on the point cloud:** Controls the horizontal spread of 3D points. If `fx` is too small, the point cloud is horizontally compressed. If too large, it is horizontally stretched.

**How to find it:** Use your camera's calibration matrix `K[0][0]`, or compute it from the horizontal field of view `FoV_h`:

```
fx = (image_width / 2) / tan(FoV_h / 2)
```

---

### `fy` — vertical focal length (pixels)

**Default:** `None` (auto-derived as `image.width × 0.85`, same as `fx`)

The vertical focal length in pixels. For cameras with square pixels, `fy ≈ fx`. Cameras with non-square sensors may have `fy ≠ fx`.

**Effect on the point cloud:** Controls vertical spread analogously to `fx`. A `fy` that does not match your sensor produces vertically compressed or stretched geometry.

**How to find it:** `K[1][1]` from the calibration matrix, or:

```
fy = (image_height / 2) / tan(FoV_v / 2)
```

---

### `cx` — horizontal principal point (pixels)

**Default:** `None` (auto-derived as `image.width / 2`)

The horizontal image coordinate of the optical axis — ideally the exact centre of the sensor.

**Effect on the point cloud:** Shifts the entire point cloud left or right. A `cx` that does not match your sensor makes the scene appear viewed from an off-centre vantage point, introducing a lateral tilt.

---

### `cy` — vertical principal point (pixels)

**Default:** `None` (auto-derived as `image.height / 2`)

The vertical image coordinate of the optical axis.

**Effect on the point cloud:** Shifts the entire point cloud up or down. Like `cx`, a value that does not match your sensor introduces a tilt — vertical in this case.

---

## Default values

| Parameter | Default | Auto-derive formula |
|---|---|---|
| `fx` | `None` | `image.width × 0.85` (~63° FOV) |
| `fy` | `None` | `image.width × 0.85` (same as fx) |
| `cx` | `None` | `image.width / 2` |
| `cy` | `None` | `image.height / 2` |

---

## Usage: Direct Python

```python
from vizion3d.lifting import (
    DepthEstimation,
    DepthEstimationAdvanceConfig,
    DepthEstimationCommand,
)

# Supply calibrated intrinsics (e.g. Intel RealSense D435 at 1280×720)
config = DepthEstimationAdvanceConfig(
    fx=909.15,
    fy=908.48,
    cx=640.0,
    cy=360.0,
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

Omit the config entirely to use auto-derived intrinsics (suitable for arbitrary photos):

```python
result = DepthEstimation().run(
    DepthEstimationCommand(
        image_input=img_bytes,
        return_point_cloud=True,
    )
)
```

---

## Usage: REST API

All four intrinsic fields are optional form fields on the `POST /lifting/depth-estimation` endpoint. Omit any field to auto-derive it from the image dimensions.

```bash
# Supply calibrated intrinsics
curl -X POST "http://localhost:8000/lifting/depth-estimation" \
  -F "image=@scene.png" \
  -F "return_point_cloud=true" \
  -F "fx=909.15" \
  -F "fy=908.48" \
  -F "cx=640.0" \
  -F "cy=360.0"
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
    },
)
data = response.json()
print(f"Depth range: {data['min_depth']:.4f} → {data['max_depth']:.4f}")
```

---

## Usage: gRPC API

The `DepthEstimationAdvanceConfig` proto message mirrors the Python model. All fields are `optional`, so any omitted field auto-derives from the image on the server side.

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
    advanced_config=lifting_pb2.DepthEstimationAdvanceConfig(
        fx=909.15,
        fy=908.48,
        cx=640.0,
        cy=360.0,
    ),
)
response = stub.RunDepthEstimation(request)
print(f"Depth range: {response.min_depth:.4f} → {response.max_depth:.4f}")
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
