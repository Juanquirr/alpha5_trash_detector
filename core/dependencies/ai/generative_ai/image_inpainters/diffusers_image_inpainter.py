from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

from core.dependencies.ai.generative_ai.image_inpainters.image_inpainter import ImageInpainter


class DiffusersImageInpainter(ImageInpainter):
    sd_model_id: str = "stabilityai/stable-diffusion-2-inpainting"
    num_inference_steps: int =50
    strength: float= 0.75
    guidance_scale: float =7.5
    prompt: str = ""
    def inpaint(self,  original_image: Image.Image, mask_image: Image.Image, prompt :str) -> Image.Image:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.sd_model_id, torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        images = pipe(
            prompt=self.prompt,
            image=original_image,
            mask_image=mask_image,
        ).images
        return images[0]