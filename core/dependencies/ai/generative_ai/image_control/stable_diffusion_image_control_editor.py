from io import BytesIO
from typing import Optional
from PIL import Image
import requests


class StableDiffusionImageControlEditor:
    def __init__(self, api_key: str, control_strength: float = 0.7, output_format: str = "png"):
        self.api_key = api_key
        self.control_strength = control_strength
        self.output_format = output_format

    def edit(self, image: Image.Image, prompt: str, negative_prompt: Optional[str] = "") -> Image.Image:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/control/structure",
            headers={
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            },
            files={"image": ("image.png", buffer.getvalue(), "image/png")},
            data={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "control_strength": str(self.control_strength),
                "output_format": self.output_format
            },
        )

        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(str(response.json()))
