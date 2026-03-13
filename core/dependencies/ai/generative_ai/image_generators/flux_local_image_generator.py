from diffusers import FluxPipeline
from PIL import Image
from pydantic import BaseModel
import torch
from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator

class FluxLocalImageGenerator(ImageGenerator, BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    _pipe: FluxPipeline = None

    def model_post_init(self, __context):
        self._pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def generate(self, prompt: str, negative_prompt: str = "") -> Image.Image:
        return self._pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=20,
            guidance_scale=3.5,
        ).images[0]
