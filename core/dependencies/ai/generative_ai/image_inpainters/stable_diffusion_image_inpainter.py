from io import BytesIO
from typing import Optional

from PIL import Image

from core.dependencies.ai.generative_ai.image_inpainters.image_inpainter import ImageInpainter


class StableDiffusionImageInpainter(ImageInpainter):
    api_key: str
    prompt: str
    negative_prompt: Optional[str] = ""
    mask_growth : int = 5

    def inpaint(self, original_image: Image.Image, mask_image: Image.Image, prompt: str) -> Image.Image:
        import requests

        original_image_buffer = BytesIO()
        mask_image_buffer = BytesIO()

        original_image.save(original_image_buffer, format="PNG")
        mask_image.save(mask_image_buffer, format="PNG")

        original_image_buffer.seek(0)
        mask_image_buffer.seek(0)

        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/edit/inpaint",
            headers={
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            },
            files={
                "image": ("original_image.png", original_image_buffer.getvalue(), "image/png"),
                "mask": ("mask_image.png", mask_image_buffer.getvalue(), "image/png"),
            },
            data={
                "prompt": prompt,
                "negative_prompt": self.negative_prompt,
                "output_format": "png",
                "grow_mask": self.mask_growth
            },
        )

        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(str(response.json()))
