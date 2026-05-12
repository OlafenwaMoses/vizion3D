from __future__ import annotations

from vizion3d.lifting.defaults import resolve_model_backend

DEFAULT_ANNOTATION_MODEL_URL = (
    "https://github.com/OlafenwaMoses/vizion3D/releases/download/essentials-v1/yolo26l-seg.pt"
)
DEFAULT_ANNOTATION_MODEL_FILENAME = "yolo26l-seg.pt"


def resolve_annotation_model_backend(
    model_backend: str,
    cache_dir=None,
) -> str:
    return resolve_model_backend(model_backend, cache_dir=cache_dir)
