from typing import Optional

from PIL import Image
import requests
from io import BytesIO
from enum import Enum, IntEnum

from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator


class StableDiffusionImageGenerator(ImageGenerator):
    class Models(Enum):
        SD3 = "sd3"
        CORE = "core"
        ULTRA = 'ultra'

    #image: Optional[Image.Image] = None
    api_key: str
    negative_prompt : Optional[str] = ""
    sd_model : Optional[Models] = Models.SD3
    aspect_ratio : Optional[str] = "1:1"

    def generate(self, prompt, negative_prompt) -> Image.Image:
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            },
            files={"none": ''},
            data={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "output_format": "png",
                "aspect_ratio": self.aspect_ratio,
                "model": "sd3.5-large",
                "steps": 30
            },
        )

        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(str(response.json()))