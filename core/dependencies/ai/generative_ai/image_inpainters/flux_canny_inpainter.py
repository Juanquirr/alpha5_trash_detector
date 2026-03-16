"""
FLUX Canny inpainter para Alpha5.

Estrategia:
1. Extrae mapa de bordes Canny de la imagen completa.
2. Borra los bordes en la región de la máscara (deja que el modelo "invente" el objeto).
3. Genera con FluxControlPipeline (FLUX.1-Canny-dev).
4. Compuesta: pega el resultado solo en la zona enmascarada sobre la imagen original.

Ventaja: el modelo respeta la estructura del fondo (agua, orillas) al generar el objeto.
"""

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel
import torch
from diffusers import FluxControlPipeline


class FluxCannyInpainter(BaseModel):
    """Inpainting aproximado vía FLUX Canny con compositing."""

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
        """Mapa Canny de la imagen con la región de máscara borrada."""
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Borrar bordes en la zona a rellenar para que el modelo genere libremente
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
        """Combina generated (zona máscara) + original (resto)."""
        orig_np = np.array(original).astype(float)
        gen_np = np.array(generated.resize(original.size, Image.LANCZOS)).astype(float)
        alpha = np.array(mask.convert("L").resize(original.size, Image.LANCZOS)).astype(float) / 255.0
        alpha = alpha[:, :, np.newaxis]  # (H, W, 1)
        result = (gen_np * alpha + orig_np * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 30.0,
    ) -> Image.Image:
        control_image = self._make_canny_control(image, mask)

        # FLUX Canny genera la imagen completa condicionada en bordes
        generated = self._pipe(
            prompt=prompt,
            control_image=control_image,
            height=image.height,
            width=image.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        # Pegar solo la región de máscara sobre el original
        return self._composite(image, generated, mask)
