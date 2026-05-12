# Camera Intrinsics Matrix

The camera intrinsics matrix **K** (also called the calibration matrix) encodes the optical properties of a camera that map 3D world points onto a 2D image plane. Every point cloud produced by vizion3d is built from these four numbers.

---

## The matrix

```
        | fx    0   cx |
K  =    |  0   fy   cy |
        |  0    0    1 |
```

In full projection form — taking a 3D point `(X, Y, Z)` in camera coordinates and projecting it to image pixel `(u, v)`:

```
| u |       | fx    0   cx |   | X/Z |
| v |  =    |  0   fy   cy | × | Y/Z |
| 1 |       |  0    0    1 |   |  1  |
```

Expanded:

```
u  =  fx × (X / Z)  +  cx
v  =  fy × (Y / Z)  +  cy
```

Inverted (what vizion3d does when building a point cloud from a depth map):

```
Z  =  d
X  =  (u - cx) × d / fx
Y  =  (v - cy) × d / fy
```

where `d` is the depth value at pixel `(u, v)`.

---

## Each element explained

### `fx` — horizontal focal length (pixels)

Position in K: row 0, col 0.

The horizontal focal length is the product of the physical focal length of the lens and the horizontal pixel density of the sensor. It describes how strongly the camera compresses horizontal depth into horizontal pixel distance.

- **Large `fx`** → narrow horizontal field of view; objects appear wider in pixel space.
- **Small `fx`** → wide horizontal field of view; objects appear narrower in pixel space.

Relation to horizontal field of view `FoV_h`:

```
fx  =  (image_width / 2) / tan(FoV_h / 2)
```

Read from a calibration matrix: K\[0\]\[0\].

---

### `fy` — vertical focal length (pixels)

Position in K: row 1, col 1.

The vertical focal length. For cameras with square pixels `fy ≈ fx`. Non-square sensors (rare in modern hardware) have `fy ≠ fx`.

- **Large `fy`** → narrow vertical field of view.
- **Small `fy`** → wide vertical field of view.

Relation to vertical field of view `FoV_v`:

```
fy  =  (image_height / 2) / tan(FoV_v / 2)
```

Read from a calibration matrix: K\[1\]\[1\].

---

### `cx` — horizontal principal point (pixels)

Position in K: row 0, col 2.

The x-coordinate of the **optical axis** on the image sensor — the pixel column where a ray travelling straight through the centre of the lens hits the sensor. Ideally the exact horizontal centre of the image.

- For a **640-wide** image: `cx ≈ 319.5`
- For a **1920-wide** image: `cx ≈ 959.5`

A miscalibrated `cx` shifts the entire point cloud left or right, making the scene appear viewed from an off-centre position.

Read from a calibration matrix: K\[0\]\[2\].

---

### `cy` — vertical principal point (pixels)

Position in K: row 1, col 2.

The y-coordinate of the optical axis on the sensor. Ideally the exact vertical centre of the image.

- For a **480-tall** image: `cy ≈ 239.5`
- For a **720-tall** image: `cy ≈ 359.5`

A miscalibrated `cy` shifts the entire point cloud up or down.

Read from a calibration matrix: K\[1\]\[2\].

---

### Off-diagonal zeros and the bottom row

```
        | fx    0   cx |
K  =    |  0   fy   cy |
        |  0    0    1 |
```

- **K\[0\]\[1\] = 0** — the skew coefficient. Zero for all modern digital cameras (pixels are rectangular).
- **K\[1\]\[0\] = 0** — symmetric constraint; always zero.
- **K\[2\]\[0\] = K\[2\]\[1\] = 0, K\[2\]\[2\] = 1** — homogeneous row. Required by the projective geometry convention; never changes.

You will never need to set these values — they are always fixed.

---

## Where to get K for your camera

| Source | How |
|---|---|
| **Camera SDK** | Most SDKs expose intrinsics directly. e.g. `intr.fx`, `intr.fy`, `intr.ppx`, `intr.ppy` on Intel RealSense. |
| **OpenCV calibration** | `cv2.calibrateCamera` returns a `camera_matrix`; read `K[0,0]`, `K[1,1]`, `K[0,2]`, `K[1,2]`. |
| **Camera datasheet** | Look for focal length in mm and sensor pixel pitch; `fx = f_mm / pixel_size_mm`. |
| **Field of view approximation** | `fx = (W/2) / tan(FoV_h/2)`, `cx = W/2 − 0.5`. Accurate enough for a first test. |

---

## Using K in vizion3d

Supply the four values via `advanced_config` on any depth or stereo command:

```python
from vizion3d.lifting import DepthEstimationAdvanceConfig

config = DepthEstimationAdvanceConfig(
    fx=909.15,   # K[0,0]
    fy=908.48,   # K[1,1]
    cx=640.0,    # K[0,2]
    cy=360.0,    # K[1,2]
)
```

Without `advanced_config`, vizion3d defaults to the **PrimeSense / Kinect v1** intrinsics at 640×480:

```
        | 525.0    0.0   319.5 |
K  =    |   0.0  525.0   239.5 |
        |   0.0    0.0     1.0 |
```

For a different camera or resolution, always supply calibrated values — intrinsics that do not match your camera produce correct topology but geometrically distorted metric scale.

See the full field reference and per-entry-point usage examples in the Advanced Config pages:

- [Depth Estimation Advanced Config](../features/depth_estimation_advanced_config.md)
