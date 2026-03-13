from PIL import Image

from core.dependencies.utils.image.mask_generators.alpha_mask_generator import AlphaMaskGenerator
from matplotlib import pyplot as plt

from core.dependencies.utils.image.mask_generators.combined_mask_generator import MaskOperator


def eliminar_pixeles_transparentes(imagen, umbral=10):
    datos = imagen.getdata()

    nuevos_datos = []
    for r, g, b, a in datos:
        if a < umbral:
            # Hacer el píxel completamente transparente
            nuevos_datos.append((0, 0, 0, 0))
        else:
            nuevos_datos.append((r, g, b, a))

    # Asignar los nuevos datos a la imagen
    imagen.putdata(nuevos_datos)
    return imagen

img = eliminar_pixeles_transparentes(Image.open("boat.png"))

mask_outside = AlphaMaskGenerator(
    alpha_image=img,
    type=AlphaMaskGenerator.Type.border_inside,
    strength=255,
    border_width=5
)
mask_inside = AlphaMaskGenerator(
    alpha_image=img,
    type=AlphaMaskGenerator.Type.border_outside,
    strength=255,
    border_width=5
)
mask_combined = (MaskOperator(resolution=img.size)
                 .combine(mask_inside)
                 .combine(mask_outside).generate())
mask_combined.save("mask2.png")
