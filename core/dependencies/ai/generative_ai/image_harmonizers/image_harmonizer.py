from abc import abstractmethod, ABC

from PIL  import Image
from pydantic import BaseModel


class ImageHarmonizer(BaseModel):
    @abstractmethod
    def harmonize(self, image : Image.Image, mask: Image.Image) -> Image.Image:
        ...
