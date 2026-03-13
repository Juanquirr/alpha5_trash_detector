from diffusers import FluxFillPipeline
from PIL import Image
import torch
from core.interfaces import ImageInpainter

class FluxInpainter(ImageInpainter):
    def __init__(self):
        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
        return self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=image.height,
            width=image.width,
            num_inference_steps=28,
            guidance_scale=30.0,
        ).images[0]
