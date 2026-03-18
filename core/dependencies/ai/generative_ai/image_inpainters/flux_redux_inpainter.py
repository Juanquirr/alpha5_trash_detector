"""
FLUX Redux + Fill inpainter.

Strategy (most powerful of the three):
  FluxPriorReduxPipeline converts a real reference image of trash
  into embeddings (prompt_embeds + pooled_prompt_embeds) in the same space
  used by FluxFillPipeline. These embeddings replace text as conditioning,
  resulting in appearance-guided inpainting from real trash images.

Expected reference structure:
  inputs/references/
    plastic_bottle/
      photo1.jpg
      photo2.jpg
    can/
      ...

If no references exist for a class, falls back to text-only FluxFill.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from pydantic import BaseModel
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline


# Explicit mapping from class_id to reference folder name.
# This avoids fuzzy matching issues (space vs underscore, partial matches).
_CLASS_FOLDER_MAP = {
    0: "plastic_bottle",
    1: "glass",
    2: "can",
    3: "plastic_bag",
    4: "metal_scrap",
    5: "plastic_wrapper",
    6: "trash_pile",
    7: "trash",
}


class FluxReduxInpainter(BaseModel):
    """Inpainting with visual guidance from reference images (Redux + Fill)."""

    model_config = {"arbitrary_types_allowed": True}
    references_dir: str = "inputs/references"

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
        """Find a random reference image for the given class."""
        folder_name = _CLASS_FOLDER_MAP.get(class_id)
        if folder_name is None:
            return None

        ref_folder = Path(self.references_dir) / folder_name
        if not ref_folder.exists() or not ref_folder.is_dir():
            return None

        candidates = [
            p
            for p in ref_folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ]

        if not candidates:
            return None

        chosen = random.choice(candidates)
        return Image.open(chosen).convert("RGB")

    def _get_embeddings(self, reference: Image.Image):
        """Extract Redux visual embeddings from the reference image."""
        prior_output = self._redux_pipe(reference)
        return prior_output.prompt_embeds, prior_output.pooled_prompt_embeds

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
        class_id: int | None = None,
    ) -> Image.Image:
        reference = self._find_reference(class_id) if class_id is not None else None

        if reference is not None:
            print(f"    [Redux] Using visual reference for class {class_id}")
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
            # No reference available: fall back to text-based Fill
            print(f"    [Redux] No reference for class {class_id}, using text prompt")
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
