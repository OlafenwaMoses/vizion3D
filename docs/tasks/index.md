# Tasks

vizion3d tasks are grouped by the kind of 3D vision work they perform. Each task
has a direct Python facade and, where available, REST and gRPC adapters.

## Lifting

| Task | Python import | REST |
|---|---|---|
| [Depth Estimation](../features/depth_estimation.md) | `vizion3d.lifting.DepthEstimation` | `/lifting/depth-estimation` |
| [Stereo Depth](../features/stereo_depth.md) | `vizion3d.stereo.StereoDepth` | `/lifting/stereo-depth` |

## Annotation

| Task | Python import | REST |
|---|---|---|
| [Object Mask Annotation 3D](../annotation/object_mask_annotation_3d.md) | `vizion3d.annotation.ObjectMaskAnnotation3D` | `/annotation/object-mask-annotation-3d` |
| [Scene Mask Annotation 3D](../annotation/scene_mask_annotation_3d.md) | `vizion3d.annotation.SceneMaskAnnotation3D` | `/annotation/scene-mask-annotation-3d` |

## Observation

| Task | Python import | REST | gRPC |
|---|---|---|---|
| [Scale Observation](../observation/scale_observation.md) | `vizion3d.observation.ScaleObservation` | `/observation/scale-observation` | `RunScaleObservation` |

## Reconstruction

| Task | Python import | REST | gRPC |
|---|---|---|---|
| [Object 3D Reconstruction](../reconstruction/object_3d_reconstruction.md) | `vizion3d.reconstruction.Object3DReconstruction` | `/reconstruction/object-3d-reconstruction` | `RunObject3DReconstruction` |
| [Scene Components 3D Reconstruction](../reconstruction/scene_components_3d_reconstruction.md) | `vizion3d.reconstruction.SceneComponents3DReconstruction` | `/reconstruction/scene-components-3d-reconstruction` | `RunSceneComponents3DReconstruction` |
