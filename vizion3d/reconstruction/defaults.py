"""Model-bundle resolution for 3D reconstruction tasks."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

MODEL_BUNDLE_FILENAME = "scene-components-3d-models.zip"
_BUNDLE_LOCK = threading.Lock()


def model_cache_dir() -> Path:
    configured = os.environ.get("VIZION3D_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "vizion3d" / "models"


def default_model_bundle() -> Path:
    configured = os.environ.get("VIZION3D_RECONSTRUCTION_MODEL_BUNDLE")
    if configured:
        return Path(configured).expanduser()
    cached = model_cache_dir() / MODEL_BUNDLE_FILENAME
    if cached.is_file():
        return cached
    checkout = Path(__file__).resolve().parents[2] / MODEL_BUNDLE_FILENAME
    return checkout


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as target:
            with urllib.request.urlopen(url) as response:
                shutil.copyfileobj(response, target)
        Path(temporary).replace(destination)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def resolve_model_bundle(value: str | os.PathLike[str] | None = None) -> Path:
    candidate = str(value or default_model_bundle())
    parsed = urllib.parse.urlparse(candidate)
    if parsed.scheme in {"http", "https"}:
        destination = model_cache_dir() / (Path(parsed.path).name or MODEL_BUNDLE_FILENAME)
        if not destination.is_file():
            with _BUNDLE_LOCK:
                if not destination.is_file():
                    _download(candidate, destination)
        return destination
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Reconstruction model bundle not found: {path}. Set "
            "VIZION3D_RECONSTRUCTION_MODEL_BUNDLE or pass model_bundle."
        )
    return path


def extract_model_bundle(value: str | os.PathLike[str] | None = None) -> Path:
    bundle = resolve_model_bundle(value)
    destination = model_cache_dir() / bundle.stem
    marker = destination / ".complete"

    def _is_current() -> bool:
        return marker.is_file() and marker.read_text(encoding="utf-8") == str(bundle)

    if _is_current():
        return destination

    with _BUNDLE_LOCK:
        if _is_current():
            return destination
        model_cache_dir().mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f"{bundle.stem}.", dir=model_cache_dir()))
        try:
            with zipfile.ZipFile(bundle) as archive:
                root = temporary.resolve()
                for member in archive.infolist():
                    target = (temporary / member.filename).resolve()
                    if root not in target.parents and target != root:
                        raise ValueError("Model bundle contains an unsafe path.")
                archive.extractall(temporary)
            if not (temporary / "TripoSR" / "model.ckpt").is_file():
                raise ValueError("Model bundle is missing TripoSR/model.ckpt.")
            if not (temporary / "TripoSR" / "config.yaml").is_file():
                raise ValueError("Model bundle is missing TripoSR/config.yaml.")
            if not (temporary / "TripoSR" / "dino-vitb16-config.json").is_file():
                raise ValueError("Model bundle is missing TripoSR/dino-vitb16-config.json.")
            if not (temporary / "TripoSR" / "tsr" / "system.py").is_file():
                raise ValueError("Model bundle is missing TripoSR/tsr/system.py.")
            if not (temporary / "rembg" / "u2net.onnx").is_file():
                raise ValueError("Model bundle is missing rembg/u2net.onnx.")
            if not (temporary / "ESRGAN" / "RealESRGAN_x4plus.pth").is_file():
                raise ValueError("Model bundle is missing ESRGAN/RealESRGAN_x4plus.pth.")
            (temporary / ".complete").write_text(str(bundle), encoding="utf-8")
            if destination.exists():
                shutil.rmtree(destination)
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return destination
