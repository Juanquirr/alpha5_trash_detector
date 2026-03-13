from io import BytesIO

from PIL import Image
import requests
from tornado.iostream import IOStream

from core.dependencies.ai.discriminative_ai.background_removers.background_remover import BackgroundRemover


class StableDiffusionBackgroundRemover(BackgroundRemover):
    api_key: str

    def remove(self, image: Image.Image) -> Image.Image:
        img = BytesIO()
        image.save(img, format="PNG")
        img.seek(0)

        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/edit/remove-background",
            headers={
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            },
            files={
                "image": ("original_image.png", img.getvalue(), "image/png")
            },
            data={
                "output_format": "webp"
            },
        )

        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(str(response.json()))
