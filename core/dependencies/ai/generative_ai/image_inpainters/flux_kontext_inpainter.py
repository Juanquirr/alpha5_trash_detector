"""
FLUX Kontext inpainter para Alpha5.

Estrategia:
  FLUX Kontext edita imágenes directamente vía instrucción de texto, sin máscara.
  Para dar control posicional se dibuja un marcador visual (elipse cian brillante)
  en la posición objetivo y se pide al modelo que lo reemplace con el objeto.

  Bbox para YOLO: diferencia de píxeles entre la imagen modificada y la original
  (sin marcador) → región que cambió = bounding box del objeto insertado.

Parámetros:
  guidance_scale recomendado para Kontext: 2.5–4.0
  num_inference_steps: 28 es suficiente
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel
import torch
from diffusers import FluxKontextPipeline


_MARKER_COLOR = (0, 255, 220)   # Cian brillante, fácilmente distinguible
_MARKER_ALPHA = 200             # Semi-transparente sobre el fondo


class FluxKontextInpainter(BaseModel):
    """Edición in-context con FLUX Kontext. No necesita máscara explícita."""

    model_config = {"arbitrary_types_allowed": True}
    diff_threshold: int = 25    # Umbral de diferencia para detectar cambios

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
        """Dibuja una elipse cian en la posición objetivo."""
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
        """Convierte un prompt descriptivo en instrucción de edición para Kontext."""
        # Quita descripciones de perspectiva que Kontext no necesita y añade la instrucción
        editing_prompt = (
            f"Replace the bright cyan ellipse marker with {prompt.lower().rstrip('.')}. "
            "Keep the rest of the image exactly as it is."
        )
        return editing_prompt

    def _compute_bbox_from_diff(
        self,
        original: np.ndarray,
        modified: np.ndarray,
    ) -> tuple | None:
        """Calcula bbox YOLO del área que cambió entre original y modificado."""
        diff = np.abs(original.astype(int) - modified.astype(int)).astype(np.uint8)
        diff_gray = diff.mean(axis=2)

        changed = (diff_gray > self.diff_threshold).astype(np.uint8)

        # Operaciones morfológicas para limpiar ruido y conectar región principal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel)
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, kernel)

        # Quedarse con el componente conectado más grande
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
        bw  = (x_max - x_min) / w
        bh  = (y_max - y_min) / h
        return x_c, y_c, bw, bh

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,          # Se usa solo para extraer cx, cy, w, h del objeto
        prompt: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.0,
    ) -> Image.Image:
        """
        Edita la imagen añadiendo el objeto.
        La máscara se usa para calcular la posición/tamaño del marcador visual.
        Devuelve (imagen_editada).
        Para obtener la bbox usa compute_bbox() después.
        """
        # Extraer centro y tamaño de la máscara
        mask_np = np.array(mask.convert("L"))
        ys, xs = np.where(mask_np > 127)
        if len(xs) == 0:
            return image

        cx = int((xs.min() + xs.max()) / 2)
        cy = int((ys.min() + ys.max()) / 2)
        obj_w = int(xs.max() - xs.min())
        obj_h = int(ys.max() - ys.min())

        # Dibujar marcador en la imagen
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
        Calcula la bbox YOLO del objeto insertado a partir de la diferencia.
        Llamar después de inpaint().
        """
        return self._compute_bbox_from_diff(
            np.array(original),
            np.array(modified),
        )
