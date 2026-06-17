# Object 3D Reconstruction

**Category:** Reconstruction
**Experimental:** No

`Object3DReconstruction` takes a close-range image of one object and produces a
cleaned, uniformly gray 3D mesh plus a uniformly gray point cloud sampled from
that mesh surface.

The task is intended for object-centric images: product shots, cropped objects,
or a single object occupying most of the frame. For broader scene images, use
[`SceneComponents3DReconstruction`](scene_components_3d_reconstruction.md).

---

<figure>
  <img src="../../assets/reconstruction/object_input.jpg" alt="Object reconstruction sample input crop" style="width:100%;border-radius:6px;">
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Sample coconut-water object input used for object reconstruction. The saved image is 1536x2048 and is capped to the default 1080px longest side inside the task.</figcaption>
</figure>

<figure>
  <div id="object-mesh-viewer" style="width:105%;margin-left:-3.5%;margin-right:-3.5%;height:440px;overflow:hidden;border-radius:6px;background:#d8d8d8;"></div>
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Generated gray object mesh: 53,965 vertices and 107,926 faces</figcaption>
</figure>

<figure>
  <div id="object-cloud-viewer" style="width:105%;margin-left:-3.5%;margin-right:-3.5%;height:440px;overflow:hidden;border-radius:6px;background:#d8d8d8;"></div>
  <figcaption style="color:#aaa;font-size:0.8em;margin-top:0.3rem;">Point cloud sampled from the generated mesh surface: 200,000 points</figcaption>
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

mountPlyViewer('object-mesh-viewer', '../../assets/reconstruction/object_sample_mesh.ply', 'mesh');
mountPlyViewer('object-cloud-viewer', '../../assets/reconstruction/object_sample_point_cloud.ply', 'points', 0.012);
</script>

---

## What It Does

The object pipeline is:

1. Load the image and cap the longest side to `max_input_dimension`.
2. Remove the background with the bundled `rembg/u2net.onnx` model.
3. Normalize the foreground image for TripoSR.
4. Run TripoSR to generate a mesh.
5. Clean the mesh, force a uniform gray material, and sample a gray point cloud.

Background removal always runs. There is no option to disable it for this task.
The output is intentionally gray so downstream workflows get geometry without
texture baking or texture-job side effects.

The sample above uses production-default mesh settings:
`marching_cubes_resolution=256`, `point_count=200000`, and
`smoothing_iterations=5`.

## Install and Models

Install the runtime dependencies:

```bash
pip install "vizion3d[reconstruction]"
```

The task resolves `scene-components-3d-models.zip` from:

1. the explicit `model_bundle` command field;
2. `VIZION3D_RECONSTRUCTION_MODEL_BUNDLE`;
3. the repository root;
4. `~/.cache/vizion3d/models`.

The bundle is extracted into the model cache and should contain:

- `ESRGAN/RealESRGAN_x4plus.pth`
- `rembg/*`
- `TripoSR/*`

Set `VIZION3D_TRIPOSR_SOURCE` when the TripoSR Python source is outside the
repository's `research/3D_Object-Reconstruction/TripoSR` directory.

## Python Usage

```python
from vizion3d.reconstruction import (
    Object3DReconstruction,
    Object3DReconstructionCommand,
    Object3DReconstructionConfig,
)

command = Object3DReconstructionCommand(
    image_input="object.png",
    model_bundle="scene-components-3d-models.zip",
    advanced_config=Object3DReconstructionConfig(
        max_input_dimension=1080,
        marching_cubes_resolution=256,
        point_count=200_000,
        device="auto",
    ),
)

result = Object3DReconstruction().run(command)

mesh = result.mesh
point_cloud = result.point_cloud
print(result.vertex_count, result.face_count, result.point_count)
```

## Command Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---:|---|---|
| `image_input` | `str \| bytes` | Yes | — | Object image path or raw image bytes. |
| `model_bundle` | `str \| None` | No | Auto-resolved | Path to `scene-components-3d-models.zip`. |
| `advanced_config` | `Object3DReconstructionConfig` | No | Defaults | Mesh, point-cloud, image-size, and device settings. |

## Config

| Field | Default | Description |
|---|---:|---|
| `max_input_dimension` | `1080` | Caps the longest source-image side before background removal. Values above `1080` are rejected. |
| `marching_cubes_resolution` | `256` | TripoSR mesh extraction resolution. Higher values can preserve more detail and cost more memory/time. |
| `density_threshold` | `25.0` | Mesh extraction density threshold. |
| `point_count` | `200000` | Number of surface points sampled from the cleaned mesh. |
| `device` | `"auto"` | Device preference. `auto` prefers available acceleration and falls back to CPU by stage. |
| `foreground_ratio` | `0.82` | Foreground normalization ratio after background removal. |
| `smoothing_iterations` | `5` | Mesh smoothing passes after reconstruction. |
| `min_component_area_ratio` | `0.02` | Removes very small disconnected mesh components. |

## Result Fields

| Field | Type | Description |
|---|---|---|
| `mesh` | `trimesh.Trimesh` | Cleaned uniformly gray mesh. |
| `point_cloud` | `open3d.geometry.PointCloud` | Uniformly gray point cloud sampled from the mesh surface. |
| `backend_used` | `str` | Resolved model-bundle extraction directory. |
| `vertex_count` | `int` | Mesh vertex count. |
| `face_count` | `int` | Mesh face count. |
| `point_count` | `int` | Sampled point-cloud size. |

## REST and gRPC Jobs

The REST and gRPC server APIs run this task as a background job because mesh
generation can take longer than a normal request timeout.

REST:

```bash
curl -X POST http://localhost:8000/reconstruction/object-3d-reconstruction \
  -F "image=@object.png" \
  -F "model_bundle=scene-components-3d-models.zip" \
  -F "device=auto"
```

The response is `201 Created`:

```json
{
  "job_id": "9f0a...",
  "status": "queued",
  "expires_at": "2026-06-16T21:00:00+00:00",
  "max_result_reads": 10,
  "result_reads_remaining": 10
}
```

Poll with:

```bash
curl http://localhost:8000/reconstruction/object-3d-reconstruction/9f0a...
```

While queued or running, polling returns `202`. When complete, it returns `200`
with:

```json
{
  "status": "succeeded",
  "result": {
    "mesh_ply": "<base64 PLY>",
    "point_cloud_ply": "<base64 PLY>",
    "vertex_count": 53965,
    "face_count": 107926,
    "point_count": 200000
  }
}
```

gRPC:

1. `RunObject3DReconstruction` submits the job and returns
   `ReconstructionJobSubmission`.
2. `GetObject3DReconstructionResult` polls by `job_id` and returns
   `Object3DReconstructionJobResponse`.

Completed results are stored in a small temp job folder on the server machine.
Set `VIZION3D_JOB_DIR` to control that folder. A result can be retrieved up to
10 times and expires after 24 hours.

## Device

`Object3DReconstructionConfig(device="auto")` propagates the selected device to
TripoSR and to `rembg` where the installed ONNX Runtime providers support it.
`auto` prefers CUDA, then Apple/CoreML-compatible acceleration for `rembg`, then
CPU. If an accelerated TripoSR or `rembg` run fails, the task retries that stage
on CPU.

## Input Resolution

The task limits the longest input-image dimension to 1080 pixels before
background removal. The resize preserves aspect ratio. This avoids spending
memory and inference time on source pixels that cannot pass through TripoSR's
final 512 by 512 conditioning input. The config may lower this limit, but
values above 1080 are rejected.

## Practical Notes

- Use object-centric images. Small or cluttered objects in a full room image are
  better handled by `SceneComponents3DReconstruction`.
- The task estimates geometry from one image; hidden backsides are inferred by
  the model and should not be treated as measured ground truth.
- Output colors are uniform gray by design. Texture generation is not part of
  this task.
