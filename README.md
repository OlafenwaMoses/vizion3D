# vizion3D

**vizion3d** is an open-source Python library for 3D computer vision — a single unified interface for depth estimation, point cloud generation, 3D object annotation, and more.

📖 **[Full documentation →](https://olafenwamoses.github.io/vizion3D/)**

---

![input image](https://olafenwamoses.github.io/vizion3D/assets/images/roomhd.jpg)

![depth map](https://olafenwamoses.github.io/vizion3D/assets/images/roomhd_depth.png)

---

## Installation

Requires **Python 3.12** (Open3D constraint).

```bash
pip install "vizion3d[cpu]"
```

For NVIDIA CUDA, AMD ROCm, or Apple Silicon MPS, see the [Hardware Acceleration](https://olafenwamoses.github.io/vizion3D/hardware_acceleration/) page.

---

## Quick start — depth estimation

```python
import open3d as o3d
from vizion3d.lifting import DepthEstimation, DepthEstimationCommand

result = DepthEstimation().run(
    DepthEstimationCommand(
        image_input="roomhd.jpg",
        return_depth_image=True,
        return_point_cloud=True,
    )
)

print(f"Depth range : {result.min_depth:.4f} → {result.max_depth:.4f}")
print(f"Points      : {len(result.point_cloud.points)}")
print(f"Scale       : {result.point_cloud_scale} metre per unit")

o3d.io.write_point_cloud("roomhd_result.ply", result.point_cloud)
```

---

## Quick start — 3D object annotation

```python
import open3d as o3d
from vizion3d.annotation import ObjectMaskAnnotation3D, ObjectMaskAnnotation3DCommand

pcd = o3d.io.read_point_cloud("roomhd_result.ply")

result = ObjectMaskAnnotation3D().run(
    ObjectMaskAnnotation3DCommand(
        point_cloud=pcd,
        image_input="roomhd.jpg",
        return_annotated_cloud=True,
    )
)

for ann in result.annotations:
    print(f"{ann.label:20s}  conf={ann.confidence:.2f}  3D points={len(ann.point_indices)}")

o3d.io.write_point_cloud("annotated.ply", result.annotated_cloud)
```

---

## Documentation

Full reference, REST/gRPC guides, hardware setup, and task catalogue:

**[https://olafenwamoses.github.io/vizion3D/](https://olafenwamoses.github.io/vizion3D/)**
