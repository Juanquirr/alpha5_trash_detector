"""
FLUX Redux + Fill inpainter para Alpha5.

Estrategia (la más potente de las tres):
  FluxPriorReduxPipeline convierte una imagen de referencia de basura real
  en embeddings (prompt_embeds + pooled_prompt_embeds) en el mismo espacio
  que usa FluxFillPipeline. Esos embeddings reemplazan el texto como
  condicionamiento → inpainting guiado por apariencia de basura real.

Estructura de referencias esperada:
  references/
    plastic bottle/  (o cualquier nombre que contenga la clase)
      foto1.jpg
      foto2.jpg
    can/
      ...

Si no hay referencias para una clase, se cae a texto puro con FluxFill normal.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline


_CLASS_NAMES = {
    0: "plastic bottle",
    1: "glass bottle",
    2: "can",
    3: "plastic bag",
    4: "metal scrap",
    5: "plastic wrapper",
    6: "trash pile",
    7: "trash",
}


class FluxReduxInpainter(BaseModel):
    """Inpainting con guía visual de imágenes de referencia (Redux + Fill)."""

    model_config = {"arbitrary_types_allowed": True}
    references_dir: str = "references"

    _redux_pipe: FluxPriorReduxPipeline = None
    _fill_pipe: FluxFillPipeline = None

    def model_post_init(self, __context):
        self._redux_pipe = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        self._fill_pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_reference(self, class_id: int) -> Image.Image | None:
        """Busca una imagen de referencia aleatoria para la clase dada."""
        class_name = _CLASS_NAMES.get(class_id, "trash")
        ref_root = Path(self.references_dir)
        if not ref_root.exists():
            return None

        # Busca carpetas que contengan el nombre de la clase (case insensitive)
        candidates: list[Path] = []
        for folder in ref_root.iterdir():
            if folder.is_dir() and class_name.lower() in folder.name.lower():
                candidates.extend(
                    p for p in folder.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                )

        if not candidates:
            return None

        chosen = random.choice(candidates)
        return Image.open(chosen).convert("RGB")

    def _get_embeddings(self, reference: Image.Image):
        """Extrae embeddings Redux de la imagen de referencia."""
        prior_output = self._redux_pipe(reference)
        return prior_output.prompt_embeds, prior_output.pooled_prompt_embeds

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        class_id: int | None = None,
    ) -> Image.Image:
        reference = self._find_reference(class_id) if class_id is not None else None

        if reference is not None:
            print(f"    [Redux] Usando referencia visual para clase {class_id}")
            prompt_embeds, pooled_prompt_embeds = self._get_embeddings(reference)

            return self._fill_pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                image=image,
                mask_image=mask,
                height=image.height,
                width=image.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=512,
            ).images[0]
        else:
            # Sin referencia: cae a Fill con texto puro
            print(f"    [Redux] Sin referencia para clase {class_id}, usando texto")
            return self._fill_pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                height=image.height,
                width=image.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=512,
            ).images[0]
