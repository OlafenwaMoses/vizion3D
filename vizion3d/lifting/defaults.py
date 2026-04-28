from __future__ import annotations

import os
import shutil
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_DEPTH_MODEL_URL = (
    "https://github.com/OlafenwaMoses/vizion3D/releases/download/"
    "essentials-v1/depth_anything_v2_vitb.pth"
)
DEFAULT_DEPTH_MODEL_FILENAME = "depth_anything_v2_vitb.pth"
DEFAULT_DEPTH_MODEL_BACKEND = "depth-anything/Depth-Anything-V2-Base-hf"
_DOWNLOAD_LOCK = threading.Lock()


def default_model_cache_dir() -> Path:
    configured = os.environ.get("VIZION3D_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "vizion3d" / "models"


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_model(url: str, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or default_model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(urllib.parse.urlparse(url).path).name or DEFAULT_DEPTH_MODEL_FILENAME
    destination = cache_dir / filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    with _DOWNLOAD_LOCK:
        if destination.exists() and destination.stat().st_size > 0:
            return destination

        fd, tmp_name = tempfile.mkstemp(prefix=f"{filename}.", suffix=".tmp", dir=cache_dir)
        try:
            with os.fdopen(fd, "wb") as tmp_file:
                with urllib.request.urlopen(url) as response:
                    shutil.copyfileobj(response, tmp_file)
            Path(tmp_name).replace(destination)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    return destination


def resolve_model_backend(model_backend: str, cache_dir: Path | None = None) -> str:
    if model_backend == DEFAULT_DEPTH_MODEL_BACKEND:
        return str(download_model(DEFAULT_DEPTH_MODEL_URL, cache_dir=cache_dir))

    if is_url(model_backend):
        return str(download_model(model_backend, cache_dir=cache_dir))

    return model_backend
