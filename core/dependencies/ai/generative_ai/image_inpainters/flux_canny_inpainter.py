"""
FLUX Canny inpainter.

Strategy:
1. Extract Canny edge map from the full image.
2. Erase edges in the mask region (let the model freely generate the object).
3. Generate with FluxControlPipeline (FLUX.1-Canny-dev).
4. Composite: paste the generated result only in the masked area over the original.

Advantage: the model respects the background structure (water, shorelines)
when generating the object.
"""

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel
import torch
from diffusers import FluxControlPipeline


class FluxCannyInpainter(BaseModel):
    """Approximate inpainting via FLUX Canny with compositing."""

    model_config = {"arbitrary_types_allowed": True}
    _pipe: FluxControlPipeline = None

    def model_post_init(self, __context):
        self._pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Canny-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_canny_control(
        self,
        image: Image.Image,
        mask: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> Image.Image:
        """Canny edge map of the image with the mask region erased."""
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Erase edges in the fill region so the model generates freely there
        mask_np = np.array(mask.convert("L"))
        edges[mask_np > 127] = 0

        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

    def _composite(
        self,
        original: Image.Image,
        generated: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        """Blend generated (mask area) + original (rest)."""
        orig_np = np.array(original).astype(float)
        gen_np = np.array(
            generated.resize(original.size, Image.LANCZOS)
        ).astype(float)
        alpha = (
            np.array(
                mask.convert("L").resize(original.size, Image.LANCZOS)
            ).astype(float)
            / 255.0
        )
        alpha = alpha[:, :, np.newaxis]  # (H, W, 1)
        result = (gen_np * alpha + orig_np * (1.0 - alpha)).clip(0, 255).astype(
            np.uint8
        )
        return Image.fromarray(result)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
    ) -> Image.Image:
        control_image = self._make_canny_control(image, mask)

        # FLUX Canny generates the full image conditioned on edges
        generated = self._pipe(
            prompt=prompt,
            control_image=control_image,
            height=image.height,
            width=image.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        # Paste only the mask region over the original
        return self._composite(image, generated, mask)
