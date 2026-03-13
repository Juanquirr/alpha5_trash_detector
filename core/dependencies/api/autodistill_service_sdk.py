import base64
import json
import requests
import hashlib
import os
from PIL import Image, ImageDraw
from io import BytesIO
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Label(BaseModel):
    id: int
    name: str
    prompt: str


class GenerateImageVariantsRequest(BaseModel):
    number_of_images: int = 1
    image: str  # base64-encoded
    annotation_model: str
    aspect_ratio: str
    prompt: str
    negative_prompt: str
    strength: float = 1.0
    labels: List[Label]


class AnnotationDetectionData(BaseModel):
    width: float
    height: float
    point: List[List[float]]

class AnnotationData(BaseModel):
    left: float
    top: float
    points: List[List[float]]

class Annotation(BaseModel):
    id: str
    type: str
    data:AnnotationData


class GenerateImageVariantsResponse(BaseModel):
    image: str  # base64-encoded result
    annotations: List[Annotation]


# ---------- SDK Client Class ----------

class AutodistillServiceClient(BaseModel):
    base_url : str
    cache_dir : str = ".segmentation_cache"

    def _hash_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return hashlib.sha256(buffered.getvalue()).hexdigest()

    def _cache_path(self, image_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{image_hash}.json")

    def _encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _decode_image(self, b64_str: str) -> Image.Image:
        if b64_str.startswith("data:image"):
            b64_str = b64_str.split(",", 1)[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data)).convert("RGB")

    def generate_segmentation(
        self,
        image: Image.Image,
        text: str = "sea",
        annotation_model: str = "grounded_sam",
        aspect_ratio: str = "square",
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 1.0,
        use_cache: bool = True
    ) -> GenerateImageVariantsResponse:
        os.makedirs(self.cache_dir, exist_ok=True)
        image_hash = self._hash_image(image)
        cache_file = self._cache_path(image_hash)

        # Return cached result if it exists
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                return GenerateImageVariantsResponse(**cached_data)

        encoded_img = self._encode_image(image)

        request_payload = GenerateImageVariantsRequest(
            number_of_images=1,
            image=encoded_img,
            annotation_model=annotation_model,
            aspect_ratio=aspect_ratio,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            labels=[Label(id=0, name=text, prompt=text)]
        )

        response = requests.post(
            f"{self.base_url}/generate_images/image_variants/",
            headers={"Content-Type": "application/json"},
            data=request_payload.json()
        )
        response.raise_for_status()

        response_data = response.json()
        result = GenerateImageVariantsResponse(**response_data)

        # Save to cache
        if use_cache:
            with open(cache_file, "w") as f:
                json.dump(result.dict(), f)

        return result

    def draw_segmentation(
        self,
        image: Image.Image,
        normalized_points: List[List[float]],
        outline_color: str = "red",
        fill_rgba: Optional[tuple] = (255, 0, 0, 80)
    ) -> Image.Image:
        """
        Draw a polygon on the image using normalized coordinates.

        Args:
            image: PIL Image.
            normalized_points: List of [x, y] normalized coordinates.
            outline_color: Outline color.
            fill_rgba: Fill color with alpha (if supported).

        Returns:
            PIL Image with polygon drawn.
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(overlay)

        width, height = image.size
        points = [(x * width, y * height) for x, y in normalized_points]
        draw.polygon(points, outline=outline_color, fill=fill_rgba)

        return Image.alpha_composite(image, overlay)

    @classmethod
    def polygon_to_bbox(cls, polygon: List[List[float]]) -> Dict[str, float]:
        """
        Converts a polygon (list of normalized [x, y] points) to a normalized bounding box.

        Args:
            polygon: List of [x, y] normalized coordinates.

        Returns:
            A dictionary with keys: 'left', 'top', 'width', 'height'
            All values are in normalized [0, 1] range.
        """
        if not polygon:
            raise ValueError("Polygon must contain at least one point.")

        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]

        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)

        return {
            "left": min_x,
            "top": min_y,
            "width": max_x - min_x,
            "height": max_y - min_y
        }
