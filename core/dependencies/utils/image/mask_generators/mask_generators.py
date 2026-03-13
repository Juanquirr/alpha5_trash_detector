from abc import abstractmethod
from PIL import Image
from pydantic import BaseModel

class MaskGenerator(BaseModel):
    @abstractmethod
    def generate(self) -> Image.Image:
        ...