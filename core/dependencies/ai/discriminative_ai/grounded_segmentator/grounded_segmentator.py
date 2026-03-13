from abc import abstractmethod, ABC
from PIL import Image
from pydantic import BaseModel

from core.dependencies.api.autodistill_service_sdk import GenerateImageVariantsResponse


class GroundedSegmentator(BaseModel, ABC):
    @abstractmethod
    def segment(self, image : Image.Image, label: str)-> GenerateImageVariantsResponse:
        ...