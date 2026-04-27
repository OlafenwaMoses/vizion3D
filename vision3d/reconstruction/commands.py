from dataclasses import dataclass
from typing import Dict

from vision3d.core.cqrs import Command

from .models import SfMResult


@dataclass
class SfMCommand(Command[SfMResult]):
    """
    Command payload to trigger a Structure from Motion (SfM) batch inference task.

    Attributes:
        images: A dictionary mapping string identifiers (like filenames) 
            to the raw bytes of the images.
        model_backend: The SfM algorithm backend to utilize. Defaults to "colmap".
    """

    images: Dict[str, bytes]  # mapping of image_id to image bytes
    model_backend: str = "colmap"
