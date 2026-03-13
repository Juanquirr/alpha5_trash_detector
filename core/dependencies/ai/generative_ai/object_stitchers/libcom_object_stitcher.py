import os
from typing import Tuple, List
from PIL import Image
from libcom import Mure_ObjectStitchModel
import torch
import tempfile

from core.dependencies.ai.generative_ai.object_stitchers.object_stitcher import ObjectStitcher


class LibcomObjectStitcher(ObjectStitcher):
    def __init__(self, device: str = 'cuda', model_type: str = 'ObjectStitch', sampler: str = 'plms'):
        self.device = torch.device(device)
        self.model = Mure_ObjectStitchModel(device=self.device, model_type=model_type, sampler=sampler)

    def save_temp_image(self, img: Image.Image, suffix: str = ".png") -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        img.save(temp_file.name)
        return temp_file.name

    def stitch(self, bg: Image.Image, fg_list: List[Image.Image], fg_mask_list: List[Image.Image],
               bbox: Tuple[int, int, int, int], num_samples: int = 1) -> Image.Image:
        # Guardar la imagen de fondo como archivo temporal
        bg_path = self.save_temp_image(bg.convert("RGB"))

        # Guardar las imágenes de primer plano y máscaras como archivos temporales
        fg_paths = [self.save_temp_image(fg.convert("RGB")) for fg in fg_list]
        fg_mask_paths = [self.save_temp_image(mask.convert("L")) for mask in fg_mask_list]

        x1, y1, x2, y2 = bbox
        bbox_list = [x1, y1, x2, y2]

        try:
            # Ejecutar el modelo con las rutas de los archivos
            comp_images, show_fg_img = self.model(
                background_image=bg_path,
                foreground_image=fg_paths,
                foreground_mask=fg_mask_paths,
                bbox=bbox_list,
                num_samples=num_samples,
                sample_steps=25,
                guidance_scale=5,
                seed=321
            )
        finally:
            # Eliminar archivos temporales
            os.remove(bg_path)
            for path in fg_paths:
                os.remove(path)
            for path in fg_mask_paths:
                os.remove(path)

        # Convertir el resultado a una imagen PIL y devolver la primera imagen generada
        result_img = Image.fromarray(comp_images)
        return result_img
