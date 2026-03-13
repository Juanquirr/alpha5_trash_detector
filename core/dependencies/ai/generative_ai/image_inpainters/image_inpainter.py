from PIL import Image
from abc import ABC, abstractmethod

from pydantic import BaseModel


class ImageInpainter(BaseModel):
    @abstractmethod
    def inpaint(self,  original_image: Image.Image, mask_image: Image.Image, prompt: str) -> Image.Image:
        ...
