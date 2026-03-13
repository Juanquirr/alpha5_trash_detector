import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import cv2

from core.dependencies.ai.generative_ai.image_inpainters.flux_local_image_inpainter import FluxLocalImageInpainter
from core.utils import divide_images_with_masks_and_bboxes, choose_random_elements

CLASSES = {
    0: "plastic bottle",
    1: "glass bottle",
    2: "can",
    3: "plastic bag",
    4: "metal scrap",
    5: "plastic wrapper",
    6: "trash pile",
    7: "trash",
}

GRID_NUMBER  = 3
INSTANCES    = 2
MASK_SIZE    = 80   # radio en píxeles del objeto insertado
INPUT_DIR    = "inputs"
OUTPUT_DIR   = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_water_cell(image: Image.Image, bbox: tuple) -> bool:
    x0, y0, x1, y1 = bbox
    crop = np.array(image.crop((x0, y0, x1, y1)))
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    avg_saturation = hsv[:, :, 1].mean()
    avg_value      = hsv[:, :, 2].mean()
    return avg_saturation > 30 and avg_value < 160


def make_small_mask(image: Image.Image, bbox: tuple, size: int = 80) -> Image.Image:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([cx - size//2, cy - size//2, cx + size//2, cy + size//2], fill=255)
    return mask


print("Cargando modelos...")
inpainter = FluxLocalImageInpainter()

image_paths = (
    list(Path(INPUT_DIR).glob("*.jpg")) +
    list(Path(INPUT_DIR).glob("*.jpeg")) +
    list(Path(INPUT_DIR).glob("*.png"))
)
print(f"Imágenes encontradas: {len(image_paths)}")

for img_path in image_paths:
    print(f"\nProcesando {img_path.name}...")
    image = Image.open(img_path).convert("RGB").resize((1024, 1024))

    # 1. Dividir en cuadrícula y filtrar solo celdas de agua
    all_cells = divide_images_with_masks_and_bboxes(image, n_divisions=GRID_NUMBER, padding=30)
    water_cells = [(mask, bbox) for mask, bbox in all_cells if is_water_cell(image, bbox)]
    print(f"  Celdas totales: {len(all_cells)} | Celdas de agua: {len(water_cells)}")

    if not water_cells:
        print(f"  ⚠️  Sin celdas de agua detectadas, saltando.")
        continue

    chosen = choose_random_elements(water_cells, min(INSTANCES, len(water_cells)))

    # 2. Insertar objetos con máscara pequeña
    annotations = []
    for _, bbox in chosen:
        class_id   = random.choice(list(CLASSES.keys()))
        class_name = CLASSES[class_id]
        prompt = (
            f"aerial view from drone, {class_name} floating on sea surface, "
            f"small object, photorealistic, ocean water background"
        )
        print(f"  → Insertando '{class_name}'...")
        small_mask = make_small_mask(image, bbox, size=MASK_SIZE)
        image = inpainter.inpaint(image, small_mask, prompt)

        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = MASK_SIZE // 2
        x_c = cx / image.width
        y_c = cy / image.height
        w   = (MASK_SIZE) / image.width
        h   = (MASK_SIZE) / image.height
        annotations.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # 3. Guardar
    stem = img_path.stem
    image.save(f"{OUTPUT_DIR}/{stem}_synth.png")
    with open(f"{OUTPUT_DIR}/{stem}_synth.txt", "w") as f:
        f.write("\n".join(annotations))
    print(f"  ✓ Guardado en outputs/{stem}_synth.png")

print("\nGeneración completada.")
