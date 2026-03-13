from abc import abstractmethod, ABC
from PIL import Image
from pydantic import BaseModel


class ImageEditor(ABC, BaseModel):
    @abstractmethod
    def edit(self, img : Image.Image) -> Image.Image:
        ...