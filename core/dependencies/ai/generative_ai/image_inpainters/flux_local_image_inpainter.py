from diffusers import FluxFillPipeline
from PIL import Image
from pydantic import BaseModel
import torch
from core.dependencies.ai.generative_ai.image_inpainters.image_inpainter import ImageInpainter


class FluxLocalImageInpainter(ImageInpainter, BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    _pipe: FluxFillPipeline = None

    def model_post_init(self, __context):
        self._pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
        result = self._pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=image.height,
            width=image.width,
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images[0]
        return result