"""Task metadata registry used by docs and service discovery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDescriptor:
    name: str
    slug: str
    category: str
    status: str
    experimental: bool
    python_import: str
    rest_endpoint: str | None
    grpc_method: str | None
    docs_path: str


PROMOTED_TASKS: tuple[TaskDescriptor, ...] = (
    TaskDescriptor(
        name="Depth Estimation",
        slug="depth-estimation",
        category="Lifting",
        status="promoted",
        experimental=False,
        python_import="vizion3d.lifting.DepthEstimation",
        rest_endpoint="/lifting/depth-estimation",
        grpc_method="RunDepthEstimation",
        docs_path="features/depth_estimation.md",
    ),
    TaskDescriptor(
        name="Stereo Depth",
        slug="stereo-depth",
        category="Lifting",
        status="promoted",
        experimental=False,
        python_import="vizion3d.stereo.StereoDepth",
        rest_endpoint="/lifting/stereo-depth",
        grpc_method="RunStereoDepth",
        docs_path="features/stereo_depth.md",
    ),
    TaskDescriptor(
        name="Object Mask Annotation 3D",
        slug="object-mask-annotation-3d",
        category="Annotation",
        status="promoted",
        experimental=False,
        python_import="vizion3d.annotation.ObjectMaskAnnotation3D",
        rest_endpoint="/annotation/object-mask-annotation-3d",
        grpc_method="RunObjectMaskAnnotation3D",
        docs_path="annotation/object_mask_annotation_3d.md",
    ),
    TaskDescriptor(
        name="Scale Observation",
        slug="scale-observation",
        category="Observation",
        status="promoted",
        experimental=False,
        python_import="vizion3d.observation.ScaleObservation",
        rest_endpoint="/observation/scale-observation",
        grpc_method="RunScaleObservation",
        docs_path="observation/scale_observation.md",
    ),
)


def tasks_by_category() -> dict[str, list[TaskDescriptor]]:
    grouped: dict[str, list[TaskDescriptor]] = {}
    for task in PROMOTED_TASKS:
        grouped.setdefault(task.category, []).append(task)
    return grouped
