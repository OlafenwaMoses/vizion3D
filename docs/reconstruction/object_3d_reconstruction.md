# Object 3D Reconstruction

`Object3DReconstruction` takes a close-range image of one object and produces:

- a cleaned, uniformly gray PLY mesh;
- a uniformly gray point cloud sampled from the mesh surface.

Install the runtime dependencies with `pip install "vizion3d[reconstruction]"`.
The task resolves `scene-components-3d-models.zip` from the repository root,
`~/.cache/vizion3d/models`, or `VIZION3D_RECONSTRUCTION_MODEL_BUNDLE`.
Set `VIZION3D_TRIPOSR_SOURCE` when the TripoSR Python source is outside the
repository's `research/3D_Object-Reconstruction/TripoSR` directory.

```python
from vizion3d.reconstruction import (
    Object3DReconstruction,
    Object3DReconstructionCommand,
)

result = Object3DReconstruction().run(
    Object3DReconstructionCommand(image_input="object.png")
)
mesh = result.mesh
point_cloud = result.point_cloud
```

Inputs always use the bundled `rembg/u2net.onnx` model for background removal.

## REST and gRPC Jobs

The REST and gRPC server APIs run this task as a background job because mesh
generation can take longer than a normal request timeout.

REST:

1. `POST /reconstruction/object-3d-reconstruction` returns `201` with a
   `job_id`.
2. `GET /reconstruction/object-3d-reconstruction/{job_id}` returns `202` while
   queued or running, then `200` with `result.mesh_ply` and
   `result.point_cloud_ply` when complete.

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
