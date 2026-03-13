from typing import Tuple
from PIL import Image
import io
import requests
from openai import OpenAI

class Dalle2ImageInpainter:
    def pil_to_io_object(self, image: Image.Image, format: str = 'PNG') -> io.BytesIO:
        io_object = io.BytesIO()
        image.save(io_object, format=format)
        io_object.seek(0)
        return io_object

    def scale_image(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        return image.resize(target_size)

    def inpaint(self, original_image: Image.Image, mask_image: Image.Image, prompt: str = "") -> Image.Image:
        client = OpenAI()
        target_size = (1024, 1024)

        # Scale images up to the required resolution
        scaled_original = self.scale_image(original_image, target_size)
        scaled_mask = self.scale_image(mask_image, target_size)

        response = client.images.edit(
            model="dall-e-2",
            image=self.pil_to_io_object(scaled_original),
            mask=self.pil_to_io_object(scaled_mask),
            prompt=prompt,
            n=1,
            size="1024x1024"
        )

        image_url = response.data[0].url
        if not image_url:
            return Image.new("RGBA", original_image.size, color="white")

        # Download the resulting image and scale it back to the original size
        image_response = requests.get(image_url)
        inpainted_image = Image.open(io.BytesIO(image_response.content))
        return self.scale_image(inpainted_image, original_image.size)
