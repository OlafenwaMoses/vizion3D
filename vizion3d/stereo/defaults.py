"""
Default model configuration and download utilities for Stereo Depth.

Model download logic is shared with the lifting module — URLs are resolved,
downloaded on first use, and cached under ``~/.cache/vizion3d/models/`` (or the
directory specified by the ``VIZION3D_MODEL_CACHE`` environment variable).
"""

from __future__ import annotations

from pathlib import Path

from vizion3d.lifting.defaults import resolve_model_backend

DEFAULT_STEREO_MODEL_URL = (
    "https://github.com/OlafenwaMoses/vizion3D/releases/download/"
    "essentials-v1/stereo-depth-s2m2-L.pth"
)
DEFAULT_STEREO_MODEL_FILENAME = "stereo-depth-s2m2-L.pth"


def resolve_stereo_model_backend(model_backend: str, cache_dir: Path | None = None) -> str:
    """Resolve *model_backend* to a local file path, downloading if needed.

    Delegates to :func:`vizion3d.lifting.defaults.resolve_model_backend` so the
    stereo and depth-estimation modules share the same download/cache logic.

    Args:
        model_backend: A URL or local file path to the S2M2 checkpoint.
        cache_dir: Override for the model cache directory.  ``None`` uses
                   ``default_model_cache_dir()`` from the lifting defaults.

    Returns:
        Absolute local path to the resolved checkpoint file.
    """
    return resolve_model_backend(model_backend, cache_dir=cache_dir)
