# Object Mask Annotation 3D

**Category:** Annotation  
**Experimental:** No

`ObjectMaskAnnotation3D` detects and instance-segments objects in a 2D RGB image, then back-projects each pixel-level segmentation mask onto the matching 3D points in a point cloud. The result is a labelled set of 3D sub-clouds — one per detected object — alongside an annotated point cloud where each object's points are recoloured with a unique colour.

A real photo is optional. When no image is provided the task synthesises a front-view RGB image by projecting the point cloud's own XYZ+RGB data into a 2D canvas; the segmentation then runs on that synthetic view.

---

<figure>
  <img src="../../assets/images/bedroom2.jpg" alt="bedroom2.jpg" style="width:100%;border-radius:6px;">
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">input image</figcaption>
</figure>

<figure>
  <div id="ply-viewer-base" style="width:105%;margin-left:-3.5%;margin-right:-3.5%;height:480px;overflow:hidden;border-radius:6px;background:#d8d8d8;"></div>
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Input point cloud generated from depth estimation</figcaption>
</figure>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';

const container = document.getElementById('ply-viewer-base');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth || 800, container.clientHeight || 480, false);
renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;';
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, (container.clientWidth || 800) / (container.clientHeight || 480), 0.001, 1000);
const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 1.0;
controls.dynamicDampingFactor = 0.2;

new ResizeObserver(() => {
  const w = renderer.domElement.clientWidth;
  const h = renderer.domElement.clientHeight;
  if (w > 0 && h > 0) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}).observe(renderer.domElement);

new PLYLoader().load('../../assets/pointclouds/bedroom2_result.ply', (geometry) => {
  const material = new THREE.PointsMaterial({ size: 0.005, vertexColors: true });
  const points = new THREE.Points(geometry, material);
  scene.add(points);
  geometry.computeBoundingBox();
  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  points.position.sub(center);
  const size = geometry.boundingBox.getSize(new THREE.Vector3()).length();
  camera.position.set(0, size * 0.3, size * 0.6);
  camera.far = size * 10;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.maxDistance = size * 5;
  controls.update();
});

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
</script>

<figure>
  <div id="ply-viewer-annotated" style="width:105%;margin-left:-3.5%;margin-right:-3.5%;height:480px;overflow:hidden;border-radius:6px;background:#d8d8d8;"></div>
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Annotated point cloud — each detected object recoloured with a unique colour</figcaption>
</figure>

<script type="module">
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';

const container = document.getElementById('ply-viewer-annotated');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth || 800, container.clientHeight || 480, false);
renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;';
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, (container.clientWidth || 800) / (container.clientHeight || 480), 0.001, 1000);
const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 1.0;
controls.dynamicDampingFactor = 0.2;

new ResizeObserver(() => {
  const w = renderer.domElement.clientWidth;
  const h = renderer.domElement.clientHeight;
  if (w > 0 && h > 0) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}).observe(renderer.domElement);

new PLYLoader().load('../../assets/pointclouds/annotated_bedroom2_result.ply', (geometry) => {
  const material = new THREE.PointsMaterial({ size: 0.008, vertexColors: true });
  const points = new THREE.Points(geometry, material);
  scene.add(points);
  geometry.computeBoundingBox();
  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  points.position.sub(center);
  const size = geometry.boundingBox.getSize(new THREE.Vector3()).length();
  camera.position.set(0, size * 0.3, size * 0.6);
  camera.far = size * 10;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.maxDistance = size * 5;
  controls.update();
});

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
</script>

Detection results from the annotated scene above:

```
chair                 conf=0.91  3D points=20806
bed                   conf=0.90  3D points=180092
tv                    conf=0.90  3D points=18410
potted plant          conf=0.53  3D points=3447
keyboard              conf=0.45  3D points=1318
vase                  conf=0.38  3D points=2321
vase                  conf=0.33  3D points=2963
potted plant          conf=0.27  3D points=15435
```

---

## Supported categories

The default checkpoint (`yolo26l-seg.pt`) is trained on the [COCO](https://cocodataset.org) dataset and can detect and segment **80 object categories**. Each detected object is identified by its `label` (string) and `class_id` (0-based integer) in the result.

| Group | Categories |
|---|---|
| **People** | person |
| **Vehicles** | bicycle, car, motorcycle, airplane, bus, train, truck, boat |
| **Outdoor** | traffic light, fire hydrant, stop sign, parking meter, bench |
| **Animals** | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| **Accessories** | backpack, umbrella, handbag, tie, suitcase |
| **Sports** | frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket |
| **Kitchen** | bottle, wine glass, cup, fork, knife, spoon, bowl |
| **Food** | banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake |
| **Furniture** | chair, couch, potted plant, bed, dining table, toilet |
| **Electronics** | tv, laptop, mouse, remote, keyboard, cell phone |
| **Appliances** | microwave, oven, toaster, sink, refrigerator |
| **Indoor** | book, clock, vase, scissors, teddy bear, hair drier, toothbrush |

Objects not in this list will not be detected. To annotate other categories, supply a custom YOLO segmentation checkpoint via `model_backend`.

---

## Model backend

Default checkpoint download:
[yolo26l-seg.pt](https://github.com/OlafenwaMoses/vizion3D/releases/download/essentials-v1/yolo26l-seg.pt)

```bash
curl -L \
  https://github.com/OlafenwaMoses/vizion3D/releases/download/essentials-v1/yolo26l-seg.pt \
  -o yolo26l-seg.pt
```

| Value | What happens |
|---|---|
| *(default)* | Downloads `yolo26l-seg.pt` to `~/.cache/vizion3d/models/` on first use, then loads it from cache |
| A local `.pt` file path | Loaded directly — never downloaded |

Models are kept in memory after the first inference in the current process. Subsequent calls to any `ObjectMaskAnnotation3D` instance reuse the loaded weights. Set `VIZION3D_MODEL_CACHE` in your environment to change the default cache directory.

---

## Command parameters

`ObjectMaskAnnotation3DCommand` is the input contract for this task.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `point_cloud` | `open3d.geometry.PointCloud` | **Yes** | — | Input point cloud in camera space (X right, Y down, Z forward), coordinates in metres. |
| `image_input` | `str \| bytes \| None` | No | `None` | RGB image to segment. Pass a file path string or raw image bytes. When `None`, a front-view image is synthesised from the point cloud automatically. |
| `model_backend` | `str` | No | vizion3D release checkpoint URL | YOLO segmentation checkpoint URL or local path. |
| `return_object_clouds` | `bool` | No | `False` | When `True`, each `MaskAnnotation3D` includes an `object_cloud` — an extracted point cloud for that object with original colours preserved. |
| `return_annotated_cloud` | `bool` | No | `False` | When `True`, the result includes a copy of the full point cloud with detected object points recoloured per object. |
| `advanced_config` | `ObjectMaskAnnotation3DConfig` | No | auto-derived from image | Camera intrinsics and detection thresholds. See [Advanced config](#advanced-config) below. |

---

## Result fields

`ObjectMaskAnnotation3DResult` is the output contract for this task.

| Field | Type | Always present | Description |
|---|---|---|---|
| `annotations` | `list[MaskAnnotation3D]` | Yes | Per-object annotations, sorted in descending confidence order. |
| `annotated_cloud` | `open3d.geometry.PointCloud \| None` | When `return_annotated_cloud=True` | Full point cloud copy with each detected object's points repainted in a unique colour. Non-object points keep their original colour. |
| `backend_used` | `str` | Yes | Resolved local file path of the YOLO checkpoint used. |

Each `MaskAnnotation3D` item contains:

| Field | Type | Description |
|---|---|---|
| `label` | `str` | COCO class name, e.g. `"person"`, `"chair"`. |
| `class_id` | `int` | COCO integer class index (0-based). |
| `confidence` | `float` | Detection confidence in `[0, 1]`. |
| `bbox_2d` | `list[float]` | Bounding box in image pixels: `[x1, y1, x2, y2]`. |
| `mask_2d` | `np.ndarray` | Boolean segmentation mask, shape `(H, W)`. |
| `point_indices` | `list[int]` | Indices into the original input point cloud for all matched 3D points. |
| `point_coords` | `list[list[float]]` | `[[x, y, z], ...]` in metres for each matched point. |
| `object_cloud` | `open3d.geometry.PointCloud \| None` | Extracted sub-cloud for this object with original colours. Present when `return_object_clouds=True`. |

---

## 1. Direct Python import — with an image

Provide an image (bytes or file path) alongside the point cloud.

```python
import open3d as o3d
from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand

pcd = o3d.io.read_point_cloud("scene.ply")

with open("scene.jpg", "rb") as f:
    img_bytes = f.read()

result = ObjectMaskAnnotation3D().run(
    ObjectMaskAnnotation3DCommand(
        point_cloud=pcd,
        image_input=img_bytes,
    )
)

print(f"Backend used : {result.backend_used}")
for ann in result.annotations:
    print(f"  {ann.label:20s}  conf={ann.confidence:.2f}  3D points={len(ann.point_indices)}")
```

---

## 2. Direct Python import — point cloud only (no image)

When `image_input` is omitted, the task synthesises a front-view RGB image directly from the point cloud's own XYZ+RGB data and runs segmentation on that synthetic view. This covers two common situations:

**No image available at all** — the point cloud came from a file, a scan, or a pipeline that did not preserve the original photo. The synthesised view is the only option.

**Stereo source with two images** — a stereo cloud is generated from a left and right image pair, but those are two separate images taken from slightly different viewpoints. There is no single image that naturally represents the combined stereo view. In this case, let the system synthesise the view from the point cloud — the synthesised view is computed from the cloud's 3D positions and stored colours, so it does not require choosing between the two frames. See [section 5](#5-stereo-point-cloud-integration) for the full stereo workflow.

The synthesised image is a point-splatting projection: each point's XYZ is projected into pixel coordinates using the camera intrinsics, and its RGB colour is painted onto a canvas. For depth-estimation clouds (one point per pixel) the result is nearly identical to the original photo. For stereo clouds or scans with variable density, sparse or occluded regions produce a patchy image that may reduce detection quality compared to a real photo.

> **Stereo clouds require explicit intrinsics.** When annotating a stereo point cloud without an image, the auto-derive heuristic cannot infer the correct focal length from the cloud geometry alone. Pass `advanced_config` with the stereo rig's actual `fx`, `fy`, `cx`, `cy` to ensure back-projection aligns masks with the 3D points. See [section 5](#5-stereo-point-cloud-integration) and [Advanced config](#advanced-config).

```python
import open3d as o3d
from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand

pcd = o3d.io.read_point_cloud("scene.ply")

result = ObjectMaskAnnotation3D().run(
    ObjectMaskAnnotation3DCommand(point_cloud=pcd)
)

for ann in result.annotations:
    print(f"{ann.label}: {len(ann.point_indices)} points")
```

---

## 3. Annotated point cloud

Request a full copy of the point cloud with each detected object recoloured in a unique colour.

```python
import open3d as o3d
from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand

pcd = o3d.io.read_point_cloud("scene.ply")

result = ObjectMaskAnnotation3D().run(
    ObjectMaskAnnotation3DCommand(
        point_cloud=pcd,
        image_input="scene.jpg",
        return_annotated_cloud=True,
    )
)

if result.annotated_cloud is not None:
    o3d.io.write_point_cloud("annotated.ply", result.annotated_cloud)
```

---

## 4. Per-object clouds

Set `return_object_clouds=True` to obtain an isolated point cloud for each detected object. Each sub-cloud uses the original colours from the input point cloud.

```python
import open3d as o3d
from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand

pcd = o3d.io.read_point_cloud("scene.ply")

result = ObjectMaskAnnotation3D().run(
    ObjectMaskAnnotation3DCommand(
        point_cloud=pcd,
        image_input="scene.jpg",
        return_object_clouds=True,
    )
)

for i, ann in enumerate(result.annotations):
    if ann.object_cloud is not None:
        path = f"object_{i:02d}_{ann.label}.ply"
        o3d.io.write_point_cloud(path, ann.object_cloud)
        print(f"Saved {path}  ({len(ann.point_indices)} points)")
```

---

## 5. Stereo point cloud integration

Point clouds produced by [Stereo Depth](../features/stereo_depth.md) are in camera space (X right, Y down, Z forward, origin at the left camera), which is exactly what this task expects. To annotate a stereo cloud correctly:

- **Always pass the stereo camera intrinsics** via `advanced_config`. The default values are for a PrimeSense sensor and will not produce back-projection that matches any other stereo rig.
- **Do not pass `image_input`** — a stereo cloud comes from two images taken at slightly different viewpoints and there is no single image that represents the combined view. Leave `image_input` unset and the system will synthesise the segmentation image directly from the point cloud's stored colours.
- **Do not centroid-shift the point cloud** before passing it in. The PLY viewer handles visual centering in JavaScript; shifting the cloud in Python breaks the Z > 0 requirement that back-projection depends on.

```python
import open3d as o3d
from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand
from vizion3d.annotation.models import ObjectMaskAnnotation3DConfig

pcd = o3d.io.read_point_cloud("stereo_result.ply")

# Intrinsics must match the stereo rig used to generate the cloud.
# Read these from your calib.txt: cam0=[fx 0 cx; 0 fy cy; 0 0 1]
stereo_cfg = ObjectMaskAnnotation3DConfig(
    fx=1733.74,
    fy=1733.74,
    cx=792.27,
    cy=541.89,
)

result = ObjectMaskAnnotation3D().run(
    ObjectMaskAnnotation3DCommand(
        point_cloud=pcd,
        return_annotated_cloud=True,
        advanced_config=stereo_cfg,
    )
)

for ann in result.annotations:
    print(f"{ann.label:20s}  conf={ann.confidence:.2f}  3D points={len(ann.point_indices)}")

o3d.io.write_point_cloud("annotated_stereo.ply", result.annotated_cloud)
```

The stereo pipeline can also generate the point cloud and annotate it in a single script:

```python
import open3d as o3d
from vizion3d.stereo import StereoDepth, StereoDepthCommand, StereoDepthAdvancedConfig
from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand
from vizion3d.annotation.models import ObjectMaskAnnotation3DConfig

# Step 1 — stereo depth → point cloud
stereo_result = StereoDepth().run(
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

# Step 2 — annotate the stereo cloud (reuse the same intrinsics)
# image_input is omitted — the system synthesises the segmentation view from the cloud.
annotation_result = ObjectMaskAnnotation3D().run(
    ObjectMaskAnnotation3DCommand(
        point_cloud=stereo_result.point_cloud,
        return_annotated_cloud=True,
        advanced_config=ObjectMaskAnnotation3DConfig(
            fx=1733.74,
            fy=1733.74,
            cx=792.27,
            cy=541.89,
        ),
    )
)

for ann in annotation_result.annotations:
    print(f"{ann.label:20s}  conf={ann.confidence:.2f}  3D points={len(ann.point_indices)}")

o3d.io.write_point_cloud("annotated_stereo.ply", annotation_result.annotated_cloud)
```

---

## 6. REST API

Start the server:

**pip / Poetry**
```bash
vizion3d-serve-rest
```

**uv**
```bash
uv run vizion3d-serve-rest
```

To preload the annotation checkpoint at startup:

```bash
uv run vizion3d-serve-rest --object_mask_annotation_3d \
  --annotation_model /models/yolo26l-seg.pt
```

Send a request with `multipart/form-data`. The `image` field is optional — omit it to let the server synthesise the front view.

**With an image:**
```bash
curl -X POST "http://localhost:8000/annotation/object-mask-annotation-3d" \
  -F "image=@scene.jpg" \
  -F "point_cloud_ply=@scene.ply" \
  -F "return_annotated_cloud=true"
```

**Point cloud only:**
```bash
curl -X POST "http://localhost:8000/annotation/object-mask-annotation-3d" \
  -F "point_cloud_ply=@scene.ply"
```

**Response** — JSON with base64-encoded binary fields:

```json
{
  "backend_used": "/path/to/yolo26l-seg.pt",
  "annotations": [
    {
      "label": "chair",
      "class_id": 56,
      "confidence": 0.87,
      "bbox_2d": [120.0, 80.0, 350.0, 420.0],
      "mask_image": "<base64-encoded PNG>",
      "point_indices": [12, 45, 103, ...],
      "object_cloud_ply": null
    }
  ],
  "annotated_cloud_ply": "<base64-encoded PLY>"
}
```

---

## 7. gRPC API

Start the server:

**pip / Poetry**
```bash
vizion3d-serve-grpc
```

**uv**
```bash
uv run vizion3d-serve-grpc
```

```python
import grpc
from vizion3d.proto import lifting_pb2, lifting_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = lifting_pb2_grpc.LiftingServiceStub(channel)

with open("scene.ply", "rb") as f:
    ply_bytes = f.read()

with open("scene.jpg", "rb") as f:
    img_bytes = f.read()

request = lifting_pb2.ObjectMaskAnnotation3DRequest(
    image_bytes=img_bytes,       # omit or leave empty for front-view synthesis
    point_cloud_ply=ply_bytes,
    return_annotated_cloud=True,
)

response = stub.RunObjectMaskAnnotation3D(request)
print(f"Backend : {response.backend_used}")
for item in response.annotations:
    print(f"  {item.label:20s}  conf={item.confidence:.2f}")
```

---

## Advanced config

`ObjectMaskAnnotation3DConfig` controls camera intrinsics and inference thresholds.

| Field | Type | Default | Description |
|---|---|---|---|
| `fx` | `float \| None` | `None` | Horizontal focal length in pixels. Auto-derived as `image_width × 0.85` when `None`. |
| `fy` | `float \| None` | `None` | Vertical focal length in pixels. Auto-derived as `image_width × 0.85` when `None`. |
| `cx` | `float \| None` | `None` | Principal point x (optical axis column). Auto-derived as `image_width / 2` when `None`. |
| `cy` | `float \| None` | `None` | Principal point y (optical axis row). Auto-derived as `image_height / 2` when `None`. |
| `conf_threshold` | `float` | `0.25` | Minimum detection confidence to keep. Range `[0, 1]`. |
| `iou_threshold` | `float` | `0.45` | Non-maximum suppression IoU overlap threshold. Range `[0, 1]`. |

When intrinsics are `None` (the default), the handler derives them from the actual image dimensions using the same `0.85 × width` field-of-view heuristic as the depth estimation pipeline. This means a point cloud generated by `DepthEstimation` can be annotated without any config — the back-projection automatically matches the intrinsics used to generate the cloud. Supply explicit values only when using a calibrated camera or a custom point cloud source. Not sure what these values are? See [Camera Intrinsics Matrix](../concepts/camera_intrinsics.md).

```python
from vizion3d.annotation import (
    ObjectMaskAnnotation3D,
    ObjectMaskAnnotation3DCommand,
    ObjectMaskAnnotation3DConfig,
)
import open3d as o3d

pcd = o3d.io.read_point_cloud("scene.ply")

result = ObjectMaskAnnotation3D().run(
    ObjectMaskAnnotation3DCommand(
        point_cloud=pcd,
        image_input="scene.jpg",
        advanced_config=ObjectMaskAnnotation3DConfig(
            fx=615.0,
            fy=615.0,
            cx=320.0,
            cy=240.0,
            conf_threshold=0.3,
        ),
    )
)
```

---

## Known limitations

- **Relative depth point clouds** — if the input point cloud was generated by monocular depth estimation (which produces relative, not metric, depth), object sizes in 3D will not correspond to real-world dimensions. For metric results, use a calibrated stereo or RGB-D camera.
- **Open3D required** — this task requires Open3D, which currently only supports Python 3.12 in this project.
- **Front-view synthesis** — when no image is supplied, the synthesised view is a simple point-splatting projection. Dense regions render well; sparse or occluded regions may produce a patchy image that reduces detection quality compared to a real photo.
