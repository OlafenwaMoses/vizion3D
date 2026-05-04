import os
import sys

# grpc_tools generates `import lifting_pb2` (bare) inside lifting_pb2_grpc.py.
# Adding this directory to sys.path makes that import resolvable when the
# package is used as `from vizion3d.proto import ...`.
sys.path.insert(0, os.path.dirname(__file__))
