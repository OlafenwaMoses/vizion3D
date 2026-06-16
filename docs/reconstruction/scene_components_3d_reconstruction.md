# Scene Components 3D Reconstruction

`SceneComponents3DReconstruction` accepts one scene image, estimates depth,
detects and segments objects, maps each mask back to the original-resolution
image, enhances each object crop with Real-ESRGAN, then runs
`Object3DReconstruction` for each selected component.

Each selected crop always goes through `rembg` background removal inside
`Object3DReconstruction`; there is no scene or object option to skip it.

Each component contains a uniformly gray mesh and point cloud, together with
its label, confidence, source bounding box, and geometry counts.

```python
from vizion3d.reconstruction import (
    SceneComponents3DReconstruction,
    SceneComponents3DReconstructionCommand,
)

result = SceneComponents3DReconstruction().run(
    SceneComponents3DReconstructionCommand(image_input="scene.jpg")
)
```

## REST and gRPC Jobs

The REST and gRPC server APIs run this task as a background job because a scene
can contain multiple object reconstructions.

REST:

1. `POST /reconstruction/scene-components-3d-reconstruction` returns `201` with
   a `job_id`.
2. `GET /reconstruction/scene-components-3d-reconstruction/{job_id}` returns
   `202` while queued or running, then `200` with reconstructed component
   meshes and point clouds under `result.components`.

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
