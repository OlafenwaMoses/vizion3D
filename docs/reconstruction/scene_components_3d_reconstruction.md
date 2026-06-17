# Scene Components 3D Reconstruction

**Category:** Reconstruction
**Experimental:** No

`SceneComponents3DReconstruction` accepts one scene image, detects objects in
the scene, crops each selected object, enhances the crop, removes the
background, and reconstructs each component as a gray mesh plus gray point
cloud.

Use this task when the input is a full scene and you want separate 3D geometry
for the detected objects. Use
[`Object3DReconstruction`](object_3d_reconstruction.md) when the image is
already a close-range view of one object.

---

<figure>
  <img src="../../assets/reconstruction/scene_components_input.jpg" alt="Scene-components reconstruction sample input" style="width:100%;border-radius:6px;">
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Sample living/dining room scene input used for component reconstruction. The saved image is 1500x1000.</figcaption>
</figure>

This sample uses the default `confidence_threshold=0.25`, analyzes the
1500x1000 input at the default 1080px scene-analysis cap, and reconstructs one
high-confidence component with production-default mesh settings:

| Component | Class ID | Confidence | Mesh vertices | Mesh faces | Point-cloud points |
|---|---:|---:|---:|---:|---:|
| couch | 57 | 0.974 | 60,344 | 120,684 | 200,000 |

<figure>
  <div id="scene-component-mesh-viewer" style="width:105%;margin-left:-3.5%;margin-right:-3.5%;height:440px;overflow:hidden;border-radius:6px;background:#d8d8d8;"></div>
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Generated gray mesh for the detected couch component</figcaption>
</figure>

<figure>
  <div id="scene-component-cloud-viewer" style="width:105%;margin-left:-3.5%;margin-right:-3.5%;height:440px;overflow:hidden;border-radius:6px;background:#d8d8d8;"></div>
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Point cloud sampled from the detected component mesh surface</figcaption>
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

function mountPlyViewer(containerId, plyPath, mode, pointSize = 0.01) {
  const container = document.getElementById(containerId);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth || 800, container.clientHeight || 440, false);
  renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;';
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 1.8));
  const key = new THREE.DirectionalLight(0xffffff, 1.8);
  key.position.set(2, 4, 3);
  scene.add(key);

  const camera = new THREE.PerspectiveCamera(55, (container.clientWidth || 800) / (container.clientHeight || 440), 0.001, 1000);
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

  new PLYLoader().load(plyPath, (geometry) => {
    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    const size = Math.max(geometry.boundingBox.getSize(new THREE.Vector3()).length(), 0.001);
    const applyRainbowColors = () => {
      const positions = geometry.getAttribute('position');
      const colors = [];
      const minY = geometry.boundingBox.min.y;
      const height = Math.max(geometry.boundingBox.max.y - minY, 0.001);
      const color = new THREE.Color();
      for (let i = 0; i < positions.count; i++) {
        const t = (positions.getY(i) - minY) / height;
        color.setHSL(0.72 * (1.0 - t), 0.95, 0.55);
        colors.push(color.r, color.g, color.b);
      }
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    };

    let object;
    if (mode === 'mesh') {
      geometry.computeVertexNormals();
      object = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({ color: 0xd3d3d3, roughness: 0.72, metalness: 0.02, side: THREE.DoubleSide })
      );
    } else {
      applyRainbowColors();
      object = new THREE.Points(
        geometry,
        new THREE.PointsMaterial({ size: pointSize, vertexColors: true })
      );
    }

    object.position.sub(center);
    scene.add(object);
    camera.position.set(0, size * 0.25, size * 0.8);
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
}

mountPlyViewer('scene-component-mesh-viewer', '../../assets/reconstruction/scene_component_01_mesh.ply', 'mesh');
mountPlyViewer('scene-component-cloud-viewer', '../../assets/reconstruction/scene_component_01_point_cloud.ply', 'points', 0.012);
</script>

---

## What It Does

The scene pipeline is:

1. Resize the image for depth and object analysis when `max_input_dimension` is
   set.
2. Estimate scene depth.
3. Detect and segment objects.
4. Map selected object masks back to the original image.
5. Crop each object with padding.
6. Enhance each crop with Real-ESRGAN.
7. Run `Object3DReconstruction` on each enhanced crop.

Each selected crop always goes through `rembg` background removal inside the
nested object reconstruction task. Real-ESRGAN is always applied for the scene
pipeline; there is no option to disable it.

The output geometry is uniformly gray. Scene components are not texture-baked.

## Install and Models

Install the runtime dependencies:

```bash
pip install "vizion3d[reconstruction]"
```

The task uses these model families:

| Stage | Default model |
|---|---|
| Depth | Depth Anything V2 |
| Detection/segmentation | YOLO segmentation backend |
| Crop enhancement | Real-ESRGAN from `scene-components-3d-models.zip` |
| Background removal | `rembg/u2net.onnx` from `scene-components-3d-models.zip` |
| Reconstruction | TripoSR from `scene-components-3d-models.zip` |

The bundled reconstruction zip is resolved the same way as
`Object3DReconstruction`: explicit `model_bundle`,
`VIZION3D_RECONSTRUCTION_MODEL_BUNDLE`, repository root, then
`~/.cache/vizion3d/models`.

## Python Usage

```python
from vizion3d.reconstruction import (
    Object3DReconstructionConfig,
    SceneComponents3DReconstruction,
    SceneComponents3DReconstructionCommand,
    SceneComponents3DReconstructionConfig,
)

command = SceneComponents3DReconstructionCommand(
    image_input="scene.jpg",
    model_bundle="scene-components-3d-models.zip",
    advanced_config=SceneComponents3DReconstructionConfig(
        max_input_dimension=1080,
        max_objects=3,
        confidence_threshold=0.25,
        padding_ratio=0.15,
        object_config=Object3DReconstructionConfig(
            max_input_dimension=1080,
            point_count=200_000,
            device="auto",
        ),
    ),
)

result = SceneComponents3DReconstruction().run(command)

for component in result.components:
    print(component.label, component.confidence, component.vertex_count)
```

## Command Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---:|---|---|
| `image_input` | `str \| bytes` | Yes | — | Scene image path or raw image bytes. |
| `model_bundle` | `str \| None` | No | Auto-resolved | Path to `scene-components-3d-models.zip`. |
| `depth_model_backend` | `str \| None` | No | Depth default | Optional depth checkpoint path or URL. |
| `annotation_model_backend` | `str \| None` | No | Annotation default | Optional segmentation checkpoint path or URL. |
| `advanced_config` | `SceneComponents3DReconstructionConfig` | No | Defaults | Scene selection and nested object reconstruction settings. |

## Config

| Field | Default | Description |
|---|---:|---|
| `max_input_dimension` | `1080` | Caps the longest scene-analysis side for depth and segmentation. Set `0` to disable only this scene-analysis resize. |
| `max_objects` | `0` | Maximum number of detected objects to reconstruct. `0` means no explicit cap. |
| `confidence_threshold` | `0.25` | Minimum detection confidence for selected components. |
| `padding_ratio` | `0.15` | Padding around each object crop before enhancement and reconstruction. |
| `object_config` | object defaults | Nested `Object3DReconstructionConfig` applied to each selected crop. |

## Result Fields

| Field | Type | Description |
|---|---|---|
| `components` | `list[SceneComponent3D]` | Reconstructed detected objects. |
| `source_image_size` | `tuple[int, int]` | Original input image size `(width, height)`. |
| `analysis_image_size` | `tuple[int, int]` | Image size used for depth and segmentation analysis. |
| `depth_backend_used` | `str` | Resolved depth backend. |
| `annotation_backend_used` | `str` | Resolved annotation backend. |
| `reconstruction_backend_used` | `str` | Resolved reconstruction model-bundle directory. |

Each component includes:

| Field | Description |
|---|---|
| `label`, `class_id`, `confidence` | Detection metadata for the selected object. |
| `bbox_2d` | Source-image box `[x1, y1, x2, y2]`. |
| `mesh` | Gray `trimesh.Trimesh`. |
| `point_cloud` | Gray `open3d.geometry.PointCloud`. |
| `vertex_count`, `face_count`, `point_count` | Geometry counts. |

## REST and gRPC Jobs

The REST and gRPC server APIs run this task as a background job because a scene
can contain multiple object reconstructions.

REST:

```bash
curl -X POST http://localhost:8000/reconstruction/scene-components-3d-reconstruction \
  -F "image=@scene.jpg" \
  -F "model_bundle=scene-components-3d-models.zip" \
  -F "max_objects=3" \
  -F "confidence_threshold=0.25" \
  -F "device=auto"
```

The response is `201 Created` with a `job_id`. Poll with:

```bash
curl http://localhost:8000/reconstruction/scene-components-3d-reconstruction/9f0a...
```

While queued or running, polling returns `202`. When complete, it returns `200`
with reconstructed component meshes and point clouds under `result.components`.
Binary PLY fields are base64-encoded in REST responses.

gRPC:

1. `RunSceneComponents3DReconstruction` submits the job and returns
   `ReconstructionJobSubmission`.
2. `GetSceneComponents3DReconstructionResult` polls by `job_id` and returns
   `SceneComponents3DReconstructionJobResponse`.

Completed results are stored in a small temp job folder on the server machine.
Set `VIZION3D_JOB_DIR` to control that folder. A result can be retrieved up to
10 times and expires after 24 hours.

## Device

The nested object config's `device` setting is propagated through the scene
pipeline. TripoSR, `rembg`, and scene Real-ESRGAN use the requested accelerator
when the installed runtime supports it, and retry on CPU if the accelerated
stage fails. Mesh cleanup and point sampling remain CPU operations because they
are handled by `trimesh`.

## Input Resolution

The scene-level `max_input_dimension=1080` applies to depth and segmentation
analysis. Object crops are still taken from the original image, enhanced with
Real-ESRGAN, then independently capped at 1080 pixels by
`Object3DReconstruction` before foreground processing. TripoSR ultimately
conditions every crop at its required 512 by 512 input size. Set the
scene-level limit to `0` to disable only the depth and segmentation resize.

## Practical Notes

- Detection quality controls what gets reconstructed. If an object is missed by
  segmentation, it will not appear in `components`.
- `max_objects` is useful for latency control. Reconstructing many scene objects
  means multiple TripoSR runs.
- Component geometry is inferred from one crop. Occluded surfaces and backsides
  are model predictions, not measured geometry.
