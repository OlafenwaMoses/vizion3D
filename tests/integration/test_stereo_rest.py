"""
Integration tests for the Stereo Depth REST entry point.

These tests exercise real S2M2 inference through POST /lifting/stereo-depth,
covering both default URL download/cache behavior and explicit local model paths.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("open3d", reason="open3d required - run: uv python pin 3.12 && uv sync")

from vizion3d.server.rest.app import app  # noqa: E402
from vizion3d.stereo.defaults import DEFAULT_STEREO_MODEL_URL  # noqa: E402
from vizion3d.stereo.handlers import StereoDepthHandler  # noqa: E402

N_RUNS = 5
COLD_LIMIT = float(os.environ.get("VIZION3D_TEST_STEREO_COLD_LIMIT", "60.0"))
WARM_LIMIT = float(os.environ.get("VIZION3D_TEST_STEREO_WARM_LIMIT", "5.0"))

client = TestClient(app, raise_server_exceptions=True)


def _save_outputs(data: dict, run_dir: Path, run: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(run_dir / f"run_{run:02d}")

    Path(prefix + ".depth_map.json").write_text(json.dumps(data["depth_map"]))
    Path(prefix + ".disparity_map.json").write_text(json.dumps(data["disparity_map"]))

    if data.get("depth_image"):
        Path(prefix + "_depth.png").write_bytes(base64.b64decode(data["depth_image"]))

    if data.get("point_cloud_ply"):
        Path(prefix + "_point_cloud.ply").write_bytes(base64.b64decode(data["point_cloud_ply"]))


def _run_group(
    model_backend: str,
    stereo_image_pair: tuple[bytes, bytes],
    stereo_advanced_config,
    run_dir: Path,
    scenario: str,
    timing_collector,
) -> list[float]:
    StereoDepthHandler._stereo_models.clear()
    left_bytes, right_bytes = stereo_image_pair
    timings: list[float] = []

    for run in range(1, N_RUNS + 1):
        t0 = time.perf_counter()
        response = client.post(
            "/lifting/stereo-depth",
            files={
                "left_image": ("left.jpg", left_bytes, "image/jpeg"),
                "right_image": ("right.jpg", right_bytes, "image/jpeg"),
            },
            data={
                "model_backend": model_backend,
                "return_depth_image": "true",
                "return_point_cloud": "true",
                "focal_length": str(stereo_advanced_config.focal_length),
                "baseline": str(stereo_advanced_config.baseline),
                "cx": str(stereo_advanced_config.cx),
                "cy": str(stereo_advanced_config.cy),
                "doffs": str(stereo_advanced_config.doffs),
                "z_far": str(stereo_advanced_config.z_far),
                "conf_threshold": str(stereo_advanced_config.conf_threshold),
                "occ_threshold": str(stereo_advanced_config.occ_threshold),
            },
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        assert response.status_code == 200, (
            f"Run {run} failed: {response.status_code} - {response.text[:300]}"
        )
        data = response.json()
        _save_outputs(data, run_dir, run)
        timing_collector.add("Stereo REST", scenario, run, elapsed, str(run_dir))

        assert isinstance(data["depth_map"], list) and len(data["depth_map"]) > 0
        assert isinstance(data["disparity_map"], list) and len(data["disparity_map"]) > 0
        assert data["max_depth"] >= data["min_depth"]
        assert data["depth_image"] is not None
        assert data["point_cloud_ply"] is not None
        assert base64.b64decode(data["depth_image"])[:4] == b"\x89PNG"
        assert base64.b64decode(data["point_cloud_ply"]).startswith(b"ply\n")

    assert len(StereoDepthHandler._stereo_models) > 0
    assert timings[0] < COLD_LIMIT
    for elapsed in timings[1:]:
        assert elapsed < WARM_LIMIT

    return timings


def test_stereo_rest_default_model(
    stereo_image_pair, stereo_advanced_config, tmp_path, timing_collector
):
    _run_group(
        model_backend=DEFAULT_STEREO_MODEL_URL,
        stereo_image_pair=stereo_image_pair,
        stereo_advanced_config=stereo_advanced_config,
        run_dir=tmp_path / "stereo_rest_default",
        scenario="Default model",
        timing_collector=timing_collector,
    )


def test_stereo_rest_local_model(
    stereo_image_pair,
    stereo_advanced_config,
    local_stereo_model_path,
    tmp_path,
    timing_collector,
):
    _run_group(
        model_backend=local_stereo_model_path,
        stereo_image_pair=stereo_image_pair,
        stereo_advanced_config=stereo_advanced_config,
        run_dir=tmp_path / "stereo_rest_local",
        scenario="Local model",
        timing_collector=timing_collector,
    )


def test_stereo_rest_advanced_config_accepted(
    stereo_image_pair, stereo_advanced_config, local_stereo_model_path
):
    StereoDepthHandler._stereo_models.clear()
    left_bytes, right_bytes = stereo_image_pair

    response = client.post(
        "/lifting/stereo-depth",
        files={
            "left_image": ("left.jpg", left_bytes, "image/jpeg"),
            "right_image": ("right.jpg", right_bytes, "image/jpeg"),
        },
        data={
            "model_backend": local_stereo_model_path,
            "return_point_cloud": "true",
            "focal_length": str(stereo_advanced_config.focal_length),
            "baseline": str(stereo_advanced_config.baseline),
            "cx": str(stereo_advanced_config.cx),
            "cy": str(stereo_advanced_config.cy),
            "doffs": str(stereo_advanced_config.doffs),
            "z_far": str(stereo_advanced_config.z_far),
            "conf_threshold": str(stereo_advanced_config.conf_threshold),
            "occ_threshold": str(stereo_advanced_config.occ_threshold),
            "scale_factor": "0.5",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["depth_map"], list) and len(data["depth_map"]) > 0
    assert data["point_cloud_ply"] is not None
    assert base64.b64decode(data["point_cloud_ply"]).startswith(b"ply\n")
