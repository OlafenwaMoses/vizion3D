# Structure from Motion (SfM)

**Category:** Reconstruction

Structure from Motion (SfM) estimates 3D structures (point clouds) and the poses of an uncalibrated camera by moving the camera across a scene or taking pictures from multiple angles.

## Features
- Process batches of 2D images.
- Generate a sparse 3D point cloud of the triangulated scene.
- Extract estimated camera poses (Rotation and Translation) for each image.

## 1. Direct Python Import

Use the `StructureFromMotion` facade class and dispatch a `SfMCommand`.

```python
from vision3d.reconstruction import StructureFromMotion, SfMCommand

# Map of string filenames to raw image bytes
images_dict = {
    "view_1.png": b"...",
    "view_2.png": b"..."
}

# Setup the command
cmd = SfMCommand(
    images=images_dict,
    model_backend="colmap"
)

# Run the task synchronously
task = StructureFromMotion()
result = task.run(cmd)

print(f"Sparse Points: {len(result.sparse_point_cloud)}")
print(f"Camera Poses: {len(result.camera_poses)}")
```

## 2. REST API (FastAPI)

The SfM task maps to a `POST /reconstruction/sfm` endpoint that accepts multiple file uploads.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/reconstruction/sfm" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "images=@view_1.png" \
  -F "images=@view_2.png" \
  -F "model_backend=colmap"
```

## 3. gRPC API

In your client, use the generated protobuf stubs:

```python
import grpc
from vision3d.proto import reconstruction_pb2, reconstruction_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
client = reconstruction_pb2_grpc.ReconstructionServiceStub(channel)

req = reconstruction_pb2.SfMRequest(
    model_backend="colmap",
    images=[
        reconstruction_pb2.ImageInput(image_id="1.png", image_bytes=b"..."),
        reconstruction_pb2.ImageInput(image_id="2.png", image_bytes=b"..."),
    ]
)

resp = client.RunSfM(req)
print(f"Points generated: {len(resp.sparse_point_cloud)}")
```
