from abc import abstractmethod, ABC
from PIL import Image
from pydantic import BaseModel


class BackgroundRemover(BaseModel):
    @abstractmethod
    def remove(self, image : Image.Image) -> Image.Image:
        ...