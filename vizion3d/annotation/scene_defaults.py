"""Defaults for the SceneMaskAnnotation3D (semantic segmentation) task."""

from __future__ import annotations

from vizion3d.lifting.defaults import resolve_model_backend

from .segformer import ADE20K_CLASSES, ADE20K_PALETTE

DEFAULT_SCENE_MODEL_URL = (
    "https://github.com/OlafenwaMoses/vizion3D/releases/download/"
    "essentials-v1/segformer_b4_ade20k.bin"
)
DEFAULT_SCENE_MODEL_FILENAME = "segformer_b4_ade20k.bin"

__all__ = [
    "ADE20K_CLASSES",
    "ADE20K_PALETTE",
    "DEFAULT_SCENE_MODEL_URL",
    "DEFAULT_SCENE_MODEL_FILENAME",
    "resolve_scene_model_backend",
]


def resolve_scene_model_backend(model_backend: str, cache_dir=None) -> str:
    return resolve_model_backend(model_backend, cache_dir=cache_dir)
