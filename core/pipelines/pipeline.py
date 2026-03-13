from abc import abstractmethod
from typing import Tuple, List, TypeVar, Generic, Dict

from PIL import Image
from pydantic import BaseModel, ConfigDict

T = TypeVar("T")

class BoundingBox(BaseModel):
    x : float
    y : float
    w : float
    h : float

class Annotation(BaseModel):
    data: Dict
    label: int
    type: str

class PipelineResult(BaseModel):
    image : Image.Image
    annotations: List[Annotation]
    model_config = ConfigDict(arbitrary_types_allowed=True)



class DatasetGenerationPipeline(BaseModel, Generic[T]):
    @abstractmethod
    def generate(self, conditioning: T) -> PipelineResult:
        ...