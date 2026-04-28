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

**pip**
```bash
pip install vizion3d
```

**Poetry**
```bash
poetry add vizion3d
```

**uv**
```bash
uv python pin 3.12
uv add vizion3d
```

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

