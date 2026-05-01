# vizion3d

**vizion3d** is an open-source Python library for 3D computer vision that gives ML/CV researchers a single, unified interface for running inference across the full spectrum of 3D vision tasks — from depth estimation and point cloud generation to NeRF reconstruction and pose estimation.

Every task is accessible through three consumption modes driven by one shared CQRS architecture:

| Mode | When to use |
|---|---|
| **Direct Python import** | Notebooks, research scripts, local prototyping |
| **REST API** | Web integrations, any-language clients |
| **gRPC API** | High-throughput, low-latency microservice pipelines |

---

## Installation

Requires **Python 3.12** (Open3D constraint).

PyTorch is **not bundled** in the base install — choose the extra that matches your hardware (see [Hardware acceleration](#hardware-acceleration) below). For most users the `cpu` extra is the right default; it installs the standard PyTorch wheel which covers CPU, NVIDIA CUDA, and Apple Silicon MPS automatically.

**pip**
```bash 
pip install "vizion3d[cpu]"
```

**Poetry**
```bash
poetry add "vizion3d[cpu]"
```

**uv**
```bash
uv python pin 3.12
uv add "vizion3d[cpu]"
```

---

## Hardware acceleration

vizion3d detects the best available device automatically at runtime — no code changes required. Choose the install extra that matches your hardware.

| Backend | Hardware | Platforms | What drives inference |
|---|---|---|---|
| **CPU** | Any processor | Linux, Windows, macOS | PyTorch CPU kernels |
| **CUDA** | NVIDIA GPU (Kepler+) | Linux, Windows | CUDA cores / Tensor Cores (Ampere+) |
| **MPS** | Apple Silicon (M1 / M2 / M3 / M4) | macOS 12.3+ | Metal GPU via unified memory |
| **ROCm** | AMD GPU (RDNA2+, CDNA, CDNA2, CDNA3) | **Linux only** | ROCm HIP runtime |

---

### CPU (default)

Works on every platform with no additional drivers. **This is the recommended install for most users.** Inference runs on PyTorch's CPU backend and automatically upgrades to NVIDIA CUDA or Apple Silicon MPS if detected at runtime — no separate install needed for those.

**pip**
```bash
pip install "vizion3d[cpu]"
```

**uv**
```bash
uv add "vizion3d[cpu]"
```

> **Mac M-series users:** the standard CPU extra automatically includes Metal Performance Shaders (MPS) support — no separate install needed. vizion3d will use your GPU via MPS as long as you are on macOS 12.3 or later with PyTorch ≥ 2.0.

---

### NVIDIA CUDA

Delivers the highest throughput for depth estimation. On NVIDIA Ampere GPUs and newer (RTX 30xx / A100 and above), PyTorch additionally uses Tensor Cores for mixed-precision acceleration.

#### Prerequisites

| Requirement | Minimum version | Link |
|---|---|---|
| NVIDIA GPU driver | 520.61.05 (Linux) / 528.33 (Windows) | [Driver downloads](https://www.nvidia.com/drivers) |
| CUDA Toolkit | 11.8 | [CUDA Toolkit installer](https://developer.nvidia.com/cuda-downloads) |
| cuDNN | 8.x | [cuDNN install guide](https://developer.nvidia.com/cudnn) |

Install CUDA and cuDNN **before** installing vizion3d. The PyTorch wheel bundled with the `cuda` extra already includes its own CUDA runtime libraries, but the driver must be present on the host.

**pip**
```bash
pip install "vizion3d[cuda]"
```

**uv**
```bash
uv add "vizion3d[cuda]"
```

vizion3d detects CUDA via `torch.cuda.is_available()` at runtime and moves models and tensors to the GPU automatically — no configuration needed.

---

### AMD ROCm

Provides GPU-accelerated inference on supported AMD GPUs using the ROCm open-source compute stack. ROCm exposes itself through PyTorch's CUDA namespace (`torch.cuda.is_available()` returns `True`), so vizion3d uses it transparently with no code changes.

> **Platform:** ROCm is supported on **Linux only**. There is no ROCm support for Windows or macOS.

#### Supported hardware

| Family | Examples |
|---|---|
| RDNA2 | RX 6700 XT, RX 6800, RX 6900 XT |
| RDNA3 | RX 7800 XT, RX 7900 XTX |
| CDNA | Instinct MI100 |
| CDNA2 | Instinct MI200 series |
| CDNA3 | Instinct MI300 series |

For the full supported GPU list see the [AMD ROCm hardware compatibility guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

#### Prerequisites

Install the ROCm stack on your system before installing the PyTorch ROCm wheel. Follow AMD's official guide:

- [ROCm installation for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)

#### Install

Because the ROCm PyTorch wheel is hosted on PyTorch's own index (not PyPI), it must be installed **before** vizion3d — vizion3d's base install has no torch dependency and will not overwrite it.

**Step 1 — install ROCm PyTorch**
```bash
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

For the full list of available ROCm wheel versions see [PyTorch ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#using-wheels-package).

**Step 2 — install vizion3d (no extra needed)**
```bash
pip install vizion3d
```

Because `vizion3d` declares no torch dependency in its base install, pip will not touch the ROCm wheel you installed in step 1.

> **Warning:** do **not** run `pip install "vizion3d[cpu]"` or `pip install "vizion3d[cuda]"` after installing the ROCm wheel — those extras pull a standard PyPI torch build and will replace your ROCm installation.

#### Limitations

- Linux only — ROCm does not run on Windows or macOS.
- Only GPUs on AMD's official support list are guaranteed to work; consumer RDNA1 cards (RX 5000 series) are not supported.
- Some PyTorch operations fall back to CPU on ROCm; performance for those ops will match CPU speed.

---

## Quick start — depth estimation

Get a depth map, point cloud, and mesh from a single image in under 10 lines.

```python
import open3d as o3d
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand

result = DepthEstimation().run(
    DepthEstimationCommand(
        image_input="scene.png",
        return_point_cloud=True,
        return_mesh=True,
    )
)

print(f"Depth range : {result.min_depth:.4f} → {result.max_depth:.4f}")
print(f"Points      : {len(result.point_cloud.points)}")
print(f"Scale       : {result.point_cloud_scale} metre per unit")

o3d.io.write_point_cloud("scene.ply", result.point_cloud)
o3d.io.write_triangle_mesh("scene_mesh.ply", result.mesh)
```

---

## Starting the servers

**pip / Poetry**
```bash
# REST API (FastAPI, default port 8000)
vizion3d-serve-rest

# gRPC API (default port 50051)
vizion3d-serve-grpc
```

**uv**
```bash
# REST API (FastAPI, default port 8000)
uv run vizion3d-serve-rest

# gRPC API (default port 50051)
uv run vizion3d-serve-grpc
```

---

## Architecture

vizion3d uses a [CQRS](https://martinfowler.com/bliki/CQRS.html) pattern throughout:

- **Commands** carry inference parameters and trigger side-effecting handlers.
- **Queries** retrieve results or metadata without side effects.
- All handlers are registered through a [`clean_ioc`](https://github.com/peter-daly/clean-ioc) container — no direct handler instantiation anywhere in the public API.

Each task lives in its own module under `vizion3d/<category>/` and exposes exactly `commands.py`, `handlers.py`, and `models.py`. Adding a new task means adding one module and one container registration — nothing else changes.

---

## Tasks

### Lifting (2D → 3D)

| Task | Status | Docs |
|---|---|---|
| Monocular depth estimation | Stable | [Depth Estimation](features/depth_estimation.md) |

