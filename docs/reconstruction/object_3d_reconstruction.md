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
