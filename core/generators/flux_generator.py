from diffusers import FluxPipeline
from PIL import Image
import torch
from core.interfaces import ImageGenerator

class FluxImageGenerator(ImageGenerator):
    def __init__(self):
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def generate(self, prompt: str, negative_prompt: str = "") -> Image.Image:
        return self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=20,
            guidance_scale=3.5,
        ).images[0]
