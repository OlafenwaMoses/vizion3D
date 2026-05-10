# Hardware Acceleration

vizion3d detects the best available device automatically at runtime — no code changes required. Choose the install path that matches your hardware.

| Backend | Hardware | Platforms | What drives inference |
|---|---|---|---|
| **CPU** | Any processor | Linux, Windows, macOS | PyTorch CPU kernels |
| **CUDA** | NVIDIA GPU (Kepler+) | Linux, Windows | CUDA cores / Tensor Cores (Ampere+) |
| **MPS** | Apple Silicon (M1 / M2 / M3 / M4) | macOS 12.3+ | Metal GPU via unified memory |
| **ROCm** | AMD GPU (RDNA2+, CDNA, CDNA2, CDNA3) | **Linux only** | ROCm HIP runtime |

---

## CPU

Works on every platform with no additional drivers.

**pip**
```bash
pip install "vizion3d[cpu]"
```

**uv**
```bash
uv add "vizion3d[cpu]"
```

---

## Apple Silicon MPS

Apple Silicon Macs (M1 and later) run inference on the GPU via Metal Performance Shaders. The `mps` extra installs the same standard PyTorch wheel as `cpu` — MPS support is built into the macOS wheel.

Requires macOS 12.3 or later.

**pip**
```bash
pip install "vizion3d[mps]"
```

**uv**
```bash
uv add "vizion3d[mps]"
```

vizion3d detects MPS via `torch.backends.mps.is_available()` at runtime and moves models to the GPU automatically. After inference, `torch.mps.empty_cache()` is called to prevent memory accumulation across repeated calls.

---

## NVIDIA CUDA

Delivers the highest throughput for depth estimation. On Ampere GPUs and newer (RTX 30xx / A100 and above), PyTorch uses Tensor Cores for mixed-precision (float16) acceleration.

### Prerequisites

| Requirement | Notes | Link |
|---|---|---|
| NVIDIA GPU driver | ≥ 450.80 (Linux) / ≥ 452.39 (Windows) — see install table below for per-version minimums | [Driver downloads](https://www.nvidia.com/drivers) |
| CUDA Toolkit | Not required for inference — the PyTorch wheel bundles its own CUDA runtime (`cudart`, `cuBLAS`, `cuDNN`, NCCL). Required only if compiling custom CUDA extensions. | — |

### Install

PyTorch CUDA wheels bundle their own CUDA runtime (`cudart`, `cuBLAS`, `cuDNN`, NCCL) — you do **not** need a matching CUDA Toolkit installed. What determines which wheel to use is your **NVIDIA driver version**:

| Wheel | Minimum driver (Linux) | Minimum driver (Windows) |
|---|---|---|
| `+cu124` (recommended) | 550.54.14 | 551.61 |
| `+cu121` | 525.60.13 | 527.41 |
| `+cu118` | 450.80.02 | 452.39 |

Use `+cu124` unless your driver is older than 550. To check: `nvidia-smi` → look at the top-right "CUDA Version" field.

Wheels must be installed **before** vizion3d using a pinned version — pinning ensures the bundled NCCL is consistent and avoids load-time `undefined symbol: ncclCommWindowDeregister` errors that occur when pip resolves torch from PyPI.

**Step 1 — install CUDA PyTorch**

*=== "driver ≥ 550 — CUDA 12.4 (recommended)"*

**pip**
```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**uv**
```bash
uv pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

*=== "driver ≥ 525 — CUDA 12.1"*

**pip**
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**uv**
```bash
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

*=== "driver ≥ 450 — CUDA 11.8"*

**pip**
```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**uv**
```bash
uv pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Step 2 — install vizion3d**

**pip**
```bash
pip install "vizion3d[cuda]"
```

**uv**
```bash
uv add "vizion3d[cuda]"
```

Because `vizion3d[cuda]` declares no torch dependency, pip will not touch the CUDA wheel installed in step 1.

vizion3d detects CUDA via `torch.cuda.is_available()` at runtime and moves models and tensors to the GPU automatically.

---

## AMD ROCm

Provides GPU-accelerated inference on supported AMD GPUs using the ROCm open-source compute stack. ROCm exposes itself through PyTorch's CUDA namespace (`torch.cuda.is_available()` returns `True`), so vizion3d uses it transparently with no code changes.

> **Platform:** ROCm is supported on **Linux only**. There is no ROCm support for Windows or macOS.

### Supported hardware

| Family | Examples |
|---|---|
| RDNA2 | RX 6700 XT, RX 6800, RX 6900 XT |
| RDNA3 | RX 7800 XT, RX 7900 XTX |
| CDNA | Instinct MI100 |
| CDNA2 | Instinct MI200 series |
| CDNA3 | Instinct MI300 series |

For the full supported GPU list see the [AMD ROCm hardware compatibility guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Prerequisites

Install the ROCm stack on your system before installing the PyTorch ROCm wheel. Follow AMD's official guide:

- [ROCm installation for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)

### Install

ROCm PyTorch wheels are hosted on PyTorch's own index, not PyPI. Install a pinned version before vizion3d.

**Step 1 — install ROCm PyTorch**

*=== "ROCm 6.2 (recommended)"*

    **pip**
```bash
pip install torch==2.5.1+rocm6.2 torchvision==0.20.1+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2
```

    **uv**
```bash
uv pip install torch==2.5.1+rocm6.2 torchvision==0.20.1+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2
```

*=== "ROCm 6.1"*

    **pip**
```bash
pip install torch==2.5.1+rocm6.1 torchvision==0.20.1+rocm6.1 --index-url https://download.pytorch.org/whl/rocm6.1
```

    **uv**
```bash
uv pip install torch==2.5.1+rocm6.1 torchvision==0.20.1+rocm6.1 --index-url https://download.pytorch.org/whl/rocm6.1
```

**Step 2 — install vizion3d**

**pip**
```bash
pip install "vizion3d[amd]"
```

**uv**
```bash
uv add "vizion3d[amd]"
```

Because `vizion3d[amd]` declares no torch dependency, pip will not touch the ROCm wheel installed in step 1.

### Limitations

- Linux only — ROCm does not run on Windows or macOS.
- Only GPUs on AMD's official support list are guaranteed to work; consumer RDNA1 cards (RX 5000 series) are not supported.
- Some PyTorch operations fall back to CPU on ROCm; performance for those ops will match CPU speed.

---

## Google Colab

Colab runtimes ship with a CUDA-enabled PyTorch pre-installed and pinned to the runtime's CUDA driver and NCCL version. Installing vizion3d with any torch extra will upgrade torch from PyPI, which can cause an NCCL symbol mismatch (`undefined symbol: ncclCommWindowDeregister`).

Install vizion3d without touching torch:

**pip**
```bash
pip install vizion3d --no-deps
pip install fastapi "clean-ioc>=1.3.0" "pydantic>=2.0.0" grpcio grpcio-tools uvicorn python-multipart "transformers>=5.6.2" "pillow>=12.2.0" "open3d>=0.18.0"
```

This installs all non-torch dependencies and leaves Colab's pre-installed torch untouched.
