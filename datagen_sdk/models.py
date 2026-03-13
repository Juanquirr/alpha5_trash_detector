from typing import Optional, List, Dict, Union

from pydantic import BaseModel,  HttpUrl

class Label(BaseModel):
    id: int
    name: str
    color: str
    order: Optional[int] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AnnotationData(BaseModel):
    point: List[float]
    width: float
    height: float


class AnnotationDetailed(BaseModel):
    id: str
    type: str
    is_synthetic: bool
    data: AnnotationData
    label: Label
    created_at: Optional[str]
    updated_at: Optional[str]






class Dataset(BaseModel):
    id: int
    name: str
    description: Optional[str]
    num_images: int
    num_labels: int
    total_weight: str
    created_at: Optional[str]
    updated_at: Optional[str]
    labels: List[Label]


class DatasetListResponse(BaseModel):
    count: int
    next: Optional[str]
    previous: Optional[str]
    results: List[Dataset]


class Annotation(BaseModel):
    data: Dict
    label: int
    type: str


class ImageItem(BaseModel):
    id: int
    image: HttpUrl
    total_weight: str
    extension: str
    name: str
    labels: List[Label]
    annotations: List[AnnotationDetailed]
    created_at: str
    updated_at: str
    is_synthetic: bool
    done: bool
    reviewed: bool
    job: Optional[str]
    batch: Optional[str]

class LabelListResponse(BaseModel):
    count: int
    next: Optional[str]
    previous: Optional[str]
    page_size: Optional[int]
    results: List[Label]

class ImageListResponse(BaseModel):
    count: int
    next: Optional[int]
    previous: Optional[int]
    page_size: int
    results: List[ImageItem]


class ImageUploadResponse(BaseModel):
    id: int
    image: str
    is_synthetic: bool
