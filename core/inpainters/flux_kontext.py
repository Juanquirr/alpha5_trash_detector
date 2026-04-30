"""
FLUX Kontext inpainter.

Strategy:
  FLUX Kontext edits images directly via text instructions, without an
  explicit mask. To provide positional control, a bright cyan ellipse marker
  is drawn at the target location and the model is instructed to replace it
  with the desired object.

  Bounding box for YOLO: computed from the pixel difference between the
  original and modified images. The largest changed region = object bbox.

Parameters:
  guidance_scale recommended for Kontext: 2.5-4.0
  num_inference_steps: 28 is sufficient
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel
import torch
from diffusers import FluxKontextPipeline


_MARKER_COLOR = (0, 255, 220)   # Bright cyan, easily distinguishable
_MARKER_ALPHA = 200             # Semi-transparent over the background


class FluxKontextInpainter(BaseModel):
    """In-context editing with FLUX Kontext. No explicit mask needed."""

    model_config = {"arbitrary_types_allowed": True}
    diff_threshold: int = 25    # Pixel difference threshold for change detection

    _pipe: FluxKontextPipeline = None

    def model_post_init(self, __context):
        self._pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _draw_marker(
        self,
        image: Image.Image,
        cx: int,
        cy: int,
        obj_w: int,
        obj_h: int,
    ) -> Image.Image:
        """Draw a cyan ellipse marker at the target position."""
        overlay = image.copy().convert("RGBA")
        draw = ImageDraw.Draw(overlay, "RGBA")
        draw.ellipse(
            [cx - obj_w // 2, cy - obj_h // 2,
             cx + obj_w // 2, cy + obj_h // 2],
            fill=(*_MARKER_COLOR, _MARKER_ALPHA),
            outline=(255, 255, 255, 255),
            width=3,
        )
        return overlay.convert("RGB")

    def _build_editing_prompt(self, prompt: str) -> str:
        """Convert a descriptive prompt into a Kontext editing instruction."""
        editing_prompt = (
            f"Replace the bright cyan ellipse marker with {prompt.lower().rstrip('.')}. "
            "The object should look naturally placed on the water surface, matching "
            "the lighting and perspective of the surrounding scene. "
            "Keep the rest of the image exactly as it is."
        )
        return editing_prompt

    def _compute_bbox_from_diff(
        self,
        original: np.ndarray,
        modified: np.ndarray,
    ) -> tuple | None:
        """Compute YOLO bbox of the area that changed between original and modified."""
        diff = np.abs(original.astype(int) - modified.astype(int)).astype(np.uint8)
        diff_gray = diff.mean(axis=2)

        changed = (diff_gray > self.diff_threshold).astype(np.uint8)

        # Morphological cleanup to connect nearby changes and remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel)
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, kernel)

        # Keep only the largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
        if num_labels <= 1:
            return None

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        region = (labels == largest).astype(np.uint8)
        ys, xs = np.where(region)
        if len(xs) == 0:
            return None

        h, w = changed.shape
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        x_c = (x_min + x_max) / 2.0 / w
        y_c = (y_min + y_max) / 2.0 / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h
        return x_c, y_c, bw, bh

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,          # Used only to extract cx, cy, w, h
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
    ) -> Image.Image:
        """
        Edit the image by adding the object at the marker position.

        The mask is used to compute the position/size for the visual marker.
        Returns the edited image.
        Call compute_bbox() afterwards to obtain the bounding box.
        """
        # Extract center and size from the mask
        mask_np = np.array(mask.convert("L"))
        ys, xs = np.where(mask_np > 127)
        if len(xs) == 0:
            return image

        cx = int((xs.min() + xs.max()) / 2)
        cy = int((ys.min() + ys.max()) / 2)
        obj_w = int(xs.max() - xs.min())
        obj_h = int(ys.max() - ys.min())

        # Draw marker and build editing instruction
        marked_image = self._draw_marker(image, cx, cy, obj_w, obj_h)
        editing_prompt = self._build_editing_prompt(prompt)

        result = self._pipe(
            image=marked_image,
            prompt=editing_prompt,
            height=image.height,
            width=image.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        return result

    def compute_bbox(
        self,
        original: Image.Image,
        modified: Image.Image,
    ) -> tuple | None:
        """
        Compute the YOLO bbox of the inserted object from pixel differences.
        Call after inpaint().
        """
        return self._compute_bbox_from_diff(
            np.array(original),
            np.array(modified),
        )
