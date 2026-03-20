"""
FLUX Fill local inpainter.

Uses FluxFillPipeline for text-conditioned mask-based inpainting.
This is the baseline model that takes an image, a mask, and a text prompt
to generate the object in the masked region.
"""

from diffusers import FluxFillPipeline
from PIL import Image
from pydantic import BaseModel
import torch
from core.inpainters.base import ImageInpainter


class FluxLocalImageInpainter(ImageInpainter, BaseModel):
    """FLUX Fill local inpainter with official model parameters."""

    model_config = {"arbitrary_types_allowed": True}
    _pipe: FluxFillPipeline = None

    def model_post_init(self, __context):
        self._pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
    ) -> Image.Image:
        return self._pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=image.height,
            width=image.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
        ).images[0]
