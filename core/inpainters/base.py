"""Base class for image inpainters."""

from abc import ABC, abstractmethod
from PIL import Image


class ImageInpainter(ABC):
    """Abstract interface for all FLUX-based inpainters."""

    @abstractmethod
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        ...
