# Scene Mask Annotation 3D

**Category:** Annotation  
**Experimental:** No

`SceneMaskAnnotation3D` runs **semantic segmentation** on a 2D RGB image, then back-projects the per-pixel class labels onto the matching 3D points in a point cloud. The result groups the cloud by **scene class** — walls, floor, ceiling, furniture, and so on — and an annotated point cloud where every point is recoloured by its class's fixed palette colour.

A real photo is optional. When no image is provided the task synthesises a front-view RGB image by projecting the point cloud's own XYZ+RGB data into a 2D canvas; the segmentation then runs on that synthetic view.

Point-cloud inputs and outputs use OpenGL/viewer camera space: `X+` right, `Y+` up, and `Z-` forward into the scene.

!!! note "This task is *semantic*, not *instance*"
    Semantic segmentation assigns **every pixel exactly one of 150 ADE20K classes**, so each 3D point belongs to exactly one class. The output therefore has **one annotation per class present** in the scene — all `wall` points are grouped into a single entry, all `floor` points into another — rather than one entry per discrete object.

---

<figure>
  <img src="../../assets/images/roomhd.jpg" alt="input image" style="width:100%;border-radius:6px;">
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">input image</figcaption>
</figure>

<figure>
  <img src="../../assets/images/scene_roomhd_mask.png" alt="semantic mask" style="width:100%;border-radius:6px;">
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Semantic mask — every pixel painted with its class's fixed colour</figcaption>
</figure>

<figure>
  <img src="../../assets/images/scene_roomhd_overlay.png" alt="labelled overlay" style="width:100%;border-radius:6px;">
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Mask overlaid on the source image, each region labelled with its class name at the region centroid</figcaption>
</figure>

<figure>
  <div id="ply-viewer-scene-base" style="width:105%;margin-left:-3.5%;margin-right:-3.5%;height:480px;overflow:hidden;border-radius:6px;background:#d8d8d8;"></div>
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

const container = document.getElementById('ply-viewer-scene-base');
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

new PLYLoader().load('../../assets/pointclouds/roomhd_result.ply', (geometry) => {
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
  <div id="ply-viewer-scene-annotated" style="width:105%;margin-left:-3.5%;margin-right:-3.5%;height:480px;overflow:hidden;border-radius:6px;background:#d8d8d8;"></div>
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Annotated point cloud — every point recoloured by its semantic class</figcaption>
</figure>

<script type="module">
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';

const container = document.getElementById('ply-viewer-scene-annotated');
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

new PLYLoader().load('../../assets/pointclouds/scene_annotated_roomhd_result.ply', (geometry) => {
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

Top classes from the annotated scene above (descending pixel count):

```
wall          pixels=947236  points=947235
floor         pixels=274227  points=274227
ceiling       pixels=246675  points=246675
curtain       pixels=204965  points=204965
bed           pixels=150265  points=150265
mirror        pixels= 39858  points= 39858
```

---

## Supported categories

The default checkpoint (`segformer_b4_ade20k.bin`) is trained on the [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset and segments **150 scene classes**. Each region is identified by its `label` (string) and `class_id` (0-based ADE20K index).

A handful of class names are aligned to their [COCO](https://cocodataset.org) synonym so masks flow directly into COCO-keyed pipelines such as [Scale Observation](../observation/scale_observation.md) size priors:

| ADE20K name | Used name (COCO) |
|---|---|
| sofa | **couch** |
| table | **dining table** |
| plant | **potted plant** |
| computer | **laptop** |
| television receiver | **tv** |
| minibike | **motorcycle** |
| glass | **wine glass** |

Ambiguous many-to-one cases keep the ADE20K name: `animal` (covers bird/cat/dog/…) and `pot`.

The full label set spans built-environment and "stuff" classes (wall, building, sky, floor, ceiling, road, sidewalk, grass, water, mountain), furniture (cabinet, chair, couch, bed, table, shelf, desk, wardrobe), and many fixtures and small objects. Classes the instance-based [Object Mask Annotation 3D](object_mask_annotation_3d.md) cannot label — walls, floor, sky, ceiling — are the core strength of this task.

---

## Model backend

Default checkpoint download:
[segformer_b4_ade20k.bin](https://github.com/OlafenwaMoses/vizion3D/releases/download/essentials-v1/segformer_b4_ade20k.bin)

```bash
curl -L \
  https://github.com/OlafenwaMoses/vizion3D/releases/download/essentials-v1/segformer_b4_ade20k.bin \
  -o segformer_b4_ade20k.bin
```

| Value | What happens |
|---|---|
| *(default)* | Downloads `segformer_b4_ade20k.bin` to `~/.cache/vizion3d/models/` on first use, then loads it from cache |
| A local `.bin` file path | Loaded directly — never downloaded |

The SegFormer architecture is vendored in `vizion3d.annotation.segformer`, so no `transformers` dependency is needed at runtime — the raw checkpoint loads directly. Models are kept in memory after the first inference in the current process. Set `VIZION3D_MODEL_CACHE` to change the cache directory.

---

## Command parameters

`SceneMaskAnnotation3DCommand` is the input contract for this task.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `point_cloud` | `open3d.geometry.PointCloud` | **Yes** | — | Input point cloud in OpenGL/viewer camera space (X right, Y up, Z negative forward), coordinates in metres. |
| `image_input` | `str \| bytes \| None` | No | `None` | RGB image to segment. Pass a file path string or raw image bytes. When `None`, a front-view image is synthesised from the point cloud automatically. |
| `model_backend` | `str` | No | vizion3D release checkpoint URL | SegFormer-B4 checkpoint URL or local path. |
| `return_region_clouds` | `bool` | No | `False` | When `True`, each `SemanticMaskAnnotation3D` includes a `region_cloud` — an extracted point cloud for that class with original colours preserved. |
| `return_annotated_cloud` | `bool` | No | `False` | When `True`, the result includes a copy of the full point cloud with every point recoloured by its class palette colour. |
| `advanced_config` | `SceneMaskAnnotation3DConfig` | No | auto-derived from image/cloud | Camera intrinsics and inference size. |

`SceneMaskAnnotation3DConfig` fields: `fx`, `fy`, `cx`, `cy` (camera intrinsics, auto-derived when `None`) and `inference_size` (shorter image edge fed to the network, default `512`; set `0` for native resolution — outputs are always at the original image resolution).

When `image_input` is omitted, the task renders a synthetic RGB front view from the point cloud before segmentation. Each point's XYZ position is projected into image pixels using `fx`, `fy`, `cx`, and `cy`, then its stored RGB colour is painted into the canvas. Pass explicit intrinsics for stereo clouds or calibrated scans; the fallback cloud-derived heuristic is useful for quick point-cloud-only runs, but it cannot recover a stereo rig's true focal length from geometry alone.

---

## Result fields

`SceneMaskAnnotation3DResult` is the output contract for this task.

| Field | Type | Always present | Description |
|---|---|---|---|
| `annotations` | `list[SemanticMaskAnnotation3D]` | Yes | Per-class annotations, sorted by descending pixel count. |
| `annotated_cloud` | `open3d.geometry.PointCloud \| None` | When `return_annotated_cloud=True` | Full point cloud copy with each point recoloured by its class palette colour. Coordinates remain OpenGL/viewer camera space. |
| `backend_used` | `str` | Yes | Resolved local file path of the SegFormer checkpoint used. |

Each `SemanticMaskAnnotation3D` item contains:

| Field | Type | Description |
|---|---|---|
| `label` | `str` | ADE20K class name (COCO-aligned where a 1:1 synonym exists). |
| `class_id` | `int` | ADE20K class index (0-based, `0..149`). |
| `bbox_2d` | `list[float]` | Axis-aligned bounding box of the class region: `[x1, y1, x2, y2]` (empty if the class has no pixels). |
| `mask_2d` | `np.ndarray` | Boolean semantic mask for this class, shape `(H, W)`. |
| `pixel_count` | `int` | Number of pixels classified as this class. |
| `point_indices` | `list[int]` | Indices into the original input point cloud for all matched 3D points. |
| `point_coords` | `list[list[float]]` | `[[x, y, z], ...]` in metres for each matched point. |
| `region_cloud` | `open3d.geometry.PointCloud \| None` | Extracted sub-cloud for this class with original colours. Present when `return_region_clouds=True`. |

---

## 1. Direct Python import

```python
import open3d as o3d
from vizion3d.annotation import SceneMaskAnnotation3D, SceneMaskAnnotation3DCommand

pcd = o3d.io.read_point_cloud("scene.ply")

with open("scene.jpg", "rb") as f:
    img_bytes = f.read()

result = SceneMaskAnnotation3D().run(
    SceneMaskAnnotation3DCommand(
        point_cloud=pcd,
        image_input=img_bytes,
        return_annotated_cloud=True,
    )
)

for ann in result.annotations:
    print(f"{ann.label:16s} pixels={ann.pixel_count:8d} points={len(ann.point_indices)}")

o3d.io.write_point_cloud("scene_annotated.ply", result.annotated_cloud)
```

Point cloud only (no image) — a front view is synthesised automatically:

```python
result = SceneMaskAnnotation3D().run(
    SceneMaskAnnotation3DCommand(point_cloud=pcd)
)
```

Point cloud only with calibrated intrinsics:

```python
from vizion3d.annotation.models import SceneMaskAnnotation3DConfig

result = SceneMaskAnnotation3D().run(
    SceneMaskAnnotation3DCommand(
        point_cloud=pcd,
        advanced_config=SceneMaskAnnotation3DConfig(
            fx=1733.74,
            fy=1733.74,
            cx=792.27,
            cy=541.89,
        ),
    )
)
```

Per-class extracted clouds:

```python
result = SceneMaskAnnotation3D().run(
    SceneMaskAnnotation3DCommand(
        point_cloud=pcd,
        image_input="scene.jpg",
        return_region_clouds=True,
    )
)
for ann in result.annotations:
    if ann.region_cloud is not None:
        o3d.io.write_point_cloud(f"region_{ann.class_id:03d}_{ann.label}.ply", ann.region_cloud)
```

---

## 2. REST API

`POST /annotation/scene-mask-annotation-3d` (multipart form).

```bash
curl -X POST http://localhost:8000/annotation/scene-mask-annotation-3d \
  -F "image=@scene.jpg" \
  -F "point_cloud_ply=@scene.ply" \
  -F "return_annotated_cloud=true"
```

Without an image (front-view synthesised from the cloud):

```bash
curl -X POST http://localhost:8000/annotation/scene-mask-annotation-3d \
  -F "point_cloud_ply=@scene.ply"
```

Form fields: `model_backend`, `return_region_clouds`, `return_annotated_cloud`, `fx`, `fy`, `cx`, `cy`, `inference_size`. The JSON response contains an `annotations` list (with base64-encoded PNG `mask_image` and optional base64 PLY `region_cloud_ply`), an optional base64 `annotated_cloud_ply`, and `backend_used`.

Start the server with the feature enabled and the model pre-loaded:

```bash
uv run vizion3d-serve-rest --scene_model /path/to/segformer_b4_ade20k.bin
```

---

## 3. gRPC

`LiftingService.RunSceneMaskAnnotation3D(SceneMaskAnnotation3DRequest) → SceneMaskAnnotation3DResponse`.

```python
import grpc
from vizion3d.proto import lifting_pb2, lifting_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = lifting_pb2_grpc.LiftingServiceStub(channel)

with open("scene.ply", "rb") as f:
    ply_bytes = f.read()
with open("scene.jpg", "rb") as f:
    img_bytes = f.read()

response = stub.RunSceneMaskAnnotation3D(
    lifting_pb2.SceneMaskAnnotation3DRequest(
        image_bytes=img_bytes,
        point_cloud_ply=ply_bytes,
        return_annotated_cloud=True,
    )
)

for item in response.annotations:
    print(item.label, item.class_id, item.pixel_count, len(item.point_indices))
```

---

## Inference performance

Warm inference (mean of 5 runs after model load) on Apple Silicon (MPS). Warm time is end-to-end per call including back-projection and mask assembly, not just the network forward pass.

| Task | Device | Input | Cold load (ms) | Warm (ms) | FPS |
|------|--------|-------|----------------|-----------|-----|
| Depth Estimation | MPS | 1000×750 | 6586 | 370 | 2.7 |
| Stereo Depth | MPS | 450×375 | 3897 | 993 | 1.0 |
| Object Mask Annotation 3D | MPS | 1000×750 | 2290 | 157 | 6.4 |
| **Scene Mask Annotation 3D** | MPS | 1000×750 | 1852 | 852 | 1.2 |
| Scale Observation | MPS | 1000×750 | 102 | 96 | 10.4 |

The SegFormer network forward pass alone is ~370 ms at `inference_size=512`; the remainder of the warm time is the per-class grouping and mask assembly over a ~2 M-point cloud. Lower `inference_size` or use a smaller dense cloud to speed this up.
