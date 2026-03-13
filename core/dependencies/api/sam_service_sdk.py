import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel


class RefineRequest(BaseModel):
    image: str  # base64 encoded
    segmentator: str
    model: str
    bboxes: List[int] = []
    points: List[int] = []


class Bbox(BaseModel):
    x_0 : int
    x_1: int
    y_0: int
    y_1: int

class Segmentation(BaseModel):
    points: List[List[int]]

class RefineResponse(BaseModel):
    bounding_boxes: List[Bbox]  # base64 encoded mask
    segmentations: List[Segmentation]


def encode_image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_str: str) -> Image.Image:
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",", 1)[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("L")


class SegmentationRefinerClient:
    def __init__(self, url: str = "http://localhost:8002/segment_image/"):
        self.url = url

    def refine_bounding_box(
        self,
        img: Image.Image,
        bbox: List[int],
        points: List[int] = [],
        segmentator: str = "sam2",
        model: str = "sam2_t.pt"
    ) -> RefineResponse:
        """
        Sends image + bbox to segmentator and returns mask + metadata.

        Returns:
            RefineResponse: Contains mask (base64) and metadata.
        """
        payload = RefineRequest(
            image=encode_image_to_base64(img),
            segmentator=segmentator,
            model=model,
            bboxes=bbox,
            points=points
        )

        response = requests.post(self.url, json=payload.dict())
        response.raise_for_status()
        return RefineResponse(**response.json())


def filter_masks_by_polygon(
    patches: List[Tuple[Image.Image]],
    polygon: List[List[float]],
    threshold: float = 0.3
) -> List[Tuple[Image.Image]]:
    """
    Filters binary masks based on their overlap with a normalized polygon.

    Args:
        patches (List[Tuple[Image.Image]]): Mask patches (e.g. from segmentation).
        polygon (List[List[float]]): Normalized [x, y] polygon points.
        threshold (float): Minimum overlap ratio.

    Returns:
        List[Tuple[Image.Image]]: Patches with sufficient overlap.
    """
    if not patches:
        return []

    width, height = patches[0][0].size
    poly_pixels = [(int(x * width), int(y * height)) for x, y in polygon]

    # Create mask from polygon
    poly_mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(poly_mask).polygon(poly_pixels, fill=1)
    poly_mask_np = np.array(poly_mask)

    filtered = []

    for patch in patches:
        patch_img = patch[0]
        patch_np = np.array(patch_img) > 0
        intersection = np.logical_and(patch_np, poly_mask_np).sum()
        patch_area = patch_np.sum()

        if patch_area == 0:
            continue

        overlap_ratio = intersection / patch_area

        if overlap_ratio >= threshold:
            filtered.append(patch)

    return filtered
