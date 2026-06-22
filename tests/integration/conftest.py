"""
Integration-test configuration.

Provides
--------
indoor_image_bytes       — real 640×480 indoor-scene JPEG for inference
stereo_image_pair        — real calibrated Middlebury Teddy stereo pair
stereo_advanced_config   — Teddy calibration mapped to StereoDepthAdvancedConfig
local_model_path         — explicit .pth path for the depth-estimation model
                           (symlinked from cache if available, else downloaded)
local_stereo_model_path  — explicit .pth path for the stereo-depth model
                           (symlinked from cache if available, else downloaded)
grpc_client_stub         — live LiftingService stub backed by an in-process
                           gRPC server running in a background thread-pool
timing_collector         — session-wide store that every test appends to
pytest_terminal_summary  — pretty inference-timing report printed at the end
"""

from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List

import grpc
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

N_RUNS = 5
ASSETS_DIR = Path(__file__).parent.parent / "assets"


# ──────────────────────────────────────────────────────────────────────────────
# Timing collector
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TimingRecord:
    task: str
    model: str
    entry_point: str
    device: str
    scenario: str
    run: int
    duration: float
    output_dir: str = ""


class InferenceTimingCollector:
    def __init__(self):
        self.records: List[TimingRecord] = []

    def add(
        self,
        entry_point: str,
        scenario: str,
        run: int,
        duration: float,
        output_dir: str = "",
        *,
        task: str | None = None,
        model: str | None = None,
        device: str | None = None,
    ) -> None:
        resolved_task, resolved_entry_point = _normalise_task_entrypoint(task, entry_point)
        self.records.append(
            TimingRecord(
                task=resolved_task,
                model=model or scenario,
                entry_point=resolved_entry_point,
                device=device or _default_device_for_task(resolved_task),
                scenario=scenario,
                run=run,
                duration=duration,
                output_dir=output_dir,
            )
        )


def _normalise_task_entrypoint(task: str | None, entry_point: str) -> tuple[str, str]:
    if task:
        return task, entry_point
    if entry_point.startswith("Scale "):
        return "Scale Observation", entry_point.removeprefix("Scale ")
    if entry_point.startswith("Stereo "):
        return "Stereo Depth", entry_point.removeprefix("Stereo ")
    return "Unknown", entry_point


def _default_device_for_task(task: str) -> str:
    if task == "Scale Observation":
        return "CPU"
    try:
        import torch
    except ImportError:
        return "CPU"
    if torch.cuda.is_available():
        return "CUDA"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS"
    return "CPU"


# Module-level singleton — pytest_terminal_summary reads from it after the session
_COLLECTOR = InferenceTimingCollector()


@pytest.fixture(scope="session")
def timing_collector() -> InferenceTimingCollector:
    return _COLLECTOR


# ──────────────────────────────────────────────────────────────────────────────
# Image fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def indoor_image_bytes() -> bytes:
    path = ASSETS_DIR / "indoor_scene.jpg"
    assert path.exists(), (
        f"Test asset not found: {path}\n"
        "Run: curl -sL 'https://picsum.photos/id/534/640/480' "
        f"-o {path}"
    )
    return path.read_bytes()


def _parse_middlebury_calibration(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key == "cam0":
            rows = raw_value.strip()[1:-1].split(";")
            matrix = [[float(v) for v in row.split()] for row in rows]
            values["focal_length"] = matrix[0][0]
            values["cx"] = matrix[0][2]
            values["cy"] = matrix[1][2]
        elif key in {"doffs", "baseline", "width", "height", "ndisp"}:
            values[key] = float(raw_value)
    return values


@pytest.fixture(scope="session")
def stereo_image_pair() -> tuple[bytes, bytes]:
    """Return a real calibrated ``(left_bytes, right_bytes)`` stereo pair.

    The fixture uses Middlebury's quarter-resolution Teddy sample. Each image is
    roughly 300 KB, and the paired ``calib.txt`` provides the camera intrinsics,
    disparity offset, and baseline used by ``stereo_advanced_config``.
    """
    left = ASSETS_DIR / "stereo" / "teddy" / "left.png"
    right = ASSETS_DIR / "stereo" / "teddy" / "right.png"
    assert left.exists(), f"Stereo left image not found: {left}"
    assert right.exists(), f"Stereo right image not found: {right}"
    return left.read_bytes(), right.read_bytes()


@pytest.fixture(scope="session")
def stereo_calibration_values() -> dict[str, float]:
    calib = ASSETS_DIR / "stereo" / "teddy" / "calib.txt"
    assert calib.exists(), f"Stereo calibration not found: {calib}"
    return _parse_middlebury_calibration(calib)


@pytest.fixture(scope="session")
def stereo_advanced_config(stereo_calibration_values):
    from vizion3d.stereo import StereoDepthAdvancedConfig

    return StereoDepthAdvancedConfig(
        focal_length=stereo_calibration_values["focal_length"],
        cx=stereo_calibration_values["cx"],
        cy=stereo_calibration_values["cy"],
        baseline=stereo_calibration_values["baseline"],
        doffs=stereo_calibration_values["doffs"],
        z_far=10.0,
        conf_threshold=0.0,
        occ_threshold=0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Local-model-path fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def local_model_path(tmp_path_factory) -> str:
    """
    Provide a .pth path in a temporary directory that is cleaned up after the
    session.  If the model is already in the default vizion3d cache we symlink
    it (free); otherwise we download it fresh.
    """
    from vizion3d.lifting.defaults import (
        DEFAULT_DEPTH_MODEL_FILENAME,
        DEFAULT_DEPTH_MODEL_URL,
        default_model_cache_dir,
        download_model,
    )

    default_cache = default_model_cache_dir() / DEFAULT_DEPTH_MODEL_FILENAME
    tmp_dir = tmp_path_factory.mktemp("local_model")
    dest = tmp_dir / DEFAULT_DEPTH_MODEL_FILENAME

    if default_cache.exists():
        dest.symlink_to(default_cache.resolve())
    else:
        download_model(DEFAULT_DEPTH_MODEL_URL, cache_dir=tmp_dir)

    assert dest.exists() or dest.is_symlink(), f"Model not found at {dest}"
    return str(dest)


@pytest.fixture(scope="session")
def local_stereo_model_path(tmp_path_factory) -> str:
    """
    Provide a local .pth path for the stereo-depth model, cleaned up after
    the session.  If the model is already in the default vizion3d cache we
    symlink it (free); otherwise we download it fresh.
    """
    from vizion3d.lifting.defaults import default_model_cache_dir, download_model
    from vizion3d.stereo.defaults import (
        DEFAULT_STEREO_MODEL_FILENAME,
        DEFAULT_STEREO_MODEL_URL,
    )

    default_cache = default_model_cache_dir() / DEFAULT_STEREO_MODEL_FILENAME
    tmp_dir = tmp_path_factory.mktemp("local_stereo_model")
    dest = tmp_dir / DEFAULT_STEREO_MODEL_FILENAME

    if default_cache.exists():
        dest.symlink_to(default_cache.resolve())
    else:
        download_model(DEFAULT_STEREO_MODEL_URL, cache_dir=tmp_dir)

    assert dest.exists() or dest.is_symlink(), f"Stereo model not found at {dest}"
    return str(dest)


@pytest.fixture(scope="session")
def local_annotation_model_path(tmp_path_factory) -> str:
    """Provide a local .pt path for the YOLO11-seg annotation model.

    Symlinks from the vizion3d cache if the model is already downloaded;
    otherwise downloads it fresh into a session-scoped tmp directory.
    """
    from vizion3d.annotation.defaults import (
        DEFAULT_ANNOTATION_MODEL_FILENAME,
        DEFAULT_ANNOTATION_MODEL_URL,
    )
    from vizion3d.lifting.defaults import default_model_cache_dir, download_model

    default_cache = default_model_cache_dir() / DEFAULT_ANNOTATION_MODEL_FILENAME
    tmp_dir = tmp_path_factory.mktemp("local_annotation_model")
    dest = tmp_dir / DEFAULT_ANNOTATION_MODEL_FILENAME

    if default_cache.exists():
        dest.symlink_to(default_cache.resolve())
    else:
        download_model(DEFAULT_ANNOTATION_MODEL_URL, cache_dir=tmp_dir)

    assert dest.exists() or dest.is_symlink(), f"Annotation model not found at {dest}"
    return str(dest)


@pytest.fixture(scope="session")
def local_scene_model_path(tmp_path_factory) -> str:
    """Provide a local .bin path for the SegFormer-B4 scene model.

    Resolution order: the vizion3d cache, then a fresh download from the release
    URL. Skips the test if neither source is available.
    """
    from vizion3d.annotation.scene_defaults import (
        DEFAULT_SCENE_MODEL_FILENAME,
        DEFAULT_SCENE_MODEL_URL,
    )
    from vizion3d.lifting.defaults import default_model_cache_dir, download_model

    default_cache = default_model_cache_dir() / DEFAULT_SCENE_MODEL_FILENAME
    tmp_dir = tmp_path_factory.mktemp("local_scene_model")
    dest = tmp_dir / DEFAULT_SCENE_MODEL_FILENAME

    if default_cache.exists():
        dest.symlink_to(default_cache.resolve())
    else:
        try:
            download_model(DEFAULT_SCENE_MODEL_URL, cache_dir=tmp_dir)
        except Exception as exc:  # pragma: no cover - network/release dependent
            pytest.skip(f"SegFormer scene model unavailable: {exc}")

    if not (dest.exists() or dest.is_symlink()):
        pytest.skip(f"Scene model not found at {dest}")
    return str(dest)


# ──────────────────────────────────────────────────────────────────────────────
# gRPC server + client stub fixture
# ──────────────────────────────────────────────────────────────────────────────

_MAX_MSG = 500 * 1024 * 1024  # match server cap

_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MSG),
    ("grpc.max_receive_message_length", _MAX_MSG),
]


@pytest.fixture(scope="session")
def grpc_client_stub():
    """
    Start a real gRPC server on a random port in a background thread pool and
    yield a connected LiftingService stub.  Server is stopped after the session.
    """
    from vizion3d.proto import lifting_pb2_grpc
    from vizion3d.server.grpc.server import LiftingServiceServicer

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=_GRPC_OPTIONS,
    )
    lifting_pb2_grpc.add_LiftingServiceServicer_to_server(LiftingServiceServicer(), server)
    port = server.add_insecure_port("[::]:0")  # 0 → OS picks a free port
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}", options=_GRPC_OPTIONS)
    stub = lifting_pb2_grpc.LiftingServiceStub(channel)

    yield stub

    channel.close()
    server.stop(grace=0)


# ──────────────────────────────────────────────────────────────────────────────
# Terminal report  (hook)
# ──────────────────────────────────────────────────────────────────────────────


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    records = _COLLECTOR.records
    if not records:
        return

    W = 120
    TASK = 28
    MODEL = 16
    EP = 10
    DEV = 7
    COLD = 17
    WARM = 17
    FPS = 13

    def _write(line: str = "") -> None:
        terminalreporter.write_line(line)

    def _thick() -> None:
        _write("━" * W)

    def _thin() -> None:
        _write(
            f"  {'─' * TASK}─┼─{'─' * MODEL}─┼─{'─' * EP}─┼─{'─' * DEV}─┼─"
            f"{'─' * COLD}─┼─{'─' * WARM}─┼─{'─' * FPS}"
        )

    def _row(task="", model="", entrypoint="", device="", cold="", warm="", fps="") -> None:
        _write(
            f"  {task:<{TASK}} │ {model:<{MODEL}} │ {entrypoint:<{EP}} │ {device:<{DEV}} │ "
            f"{cold:>{COLD}} │ {warm:>{WARM}} │ {fps:>{FPS}}"
        )

    _write()
    _thick()
    _write(f"  {'VIZION3D  ·  INTEGRATION INFERENCE TIMING REPORT':^{W - 4}}")
    _thick()
    _write()
    _row(
        "Task",
        "Model",
        "Entrypoint",
        "Device",
        "Cold run duration",
        "Warm run duration",
        "Estimated FPS",
    )
    _thin()

    def sort_key(r):
        return (r.task, r.model, r.entry_point, r.device, r.run)

    def group_key(r):
        return (r.task, r.model, r.entry_point, r.device)

    first_loads: list[float] = []
    warm_times: list[float] = []

    sorted_records = sorted(records, key=sort_key)
    groups = [(k, list(v)) for k, v in groupby(sorted_records, key=group_key)]

    for g_idx, ((task, model, entrypoint, device), recs) in enumerate(groups):
        recs = sorted(recs, key=lambda r: r.run)
        cold_runs = [r.duration for r in recs if r.run == 1]
        warm_runs = [r.duration for r in recs if r.run != 1]
        cold = min(cold_runs) if cold_runs else recs[0].duration
        warm = sum(warm_runs) / len(warm_runs) if warm_runs else 0.0
        fps = 1.0 / warm if warm > 0 else 0.0
        first_loads.append(cold)
        if warm_runs:
            warm_times.extend(warm_runs)

        if g_idx > 0:
            _thin()
        _row(
            task,
            model,
            entrypoint,
            device,
            f"{cold:.3f}s",
            f"{warm:.3f}s" if warm_runs else "n/a",
            f"{fps:.2f}" if warm_runs else "n/a",
        )

    _write()
    _thick()
    _write()

    if first_loads and warm_times:
        avg_load = sum(first_loads) / len(first_loads)
        avg_warm = sum(warm_times) / len(warm_times)
        speedup = avg_load / avg_warm if avg_warm > 0 else float("inf")
        total = len(records)

        pad = 42
        _write(f"  {'SUMMARY'}")
        _write(f"  {'─' * 58}")
        _write(f"  {'Average cold-load time':<{pad}}: {avg_load:>7.3f}s  (disk → memory)")
        _write(f"  {'Average warm inference':<{pad}}: {avg_warm:>7.3f}s  (model already in RAM)")
        _write(f"  {'In-memory speedup':<{pad}}: {speedup:>6.1f}×")
        _write(f"  {'Total inference runs':<{pad}}: {total}")

        out_dirs = sorted({r.output_dir for r in records if r.output_dir})
        if out_dirs:
            _write(f"  {'Output saved to':<{pad}}:")
            for d in out_dirs:
                _write(f"      {d}")

    _write()
    _thick()
    _write()
