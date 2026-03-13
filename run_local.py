import os
import random
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import cv2

from core.dependencies.ai.generative_ai.image_inpainters.flux_local_image_inpainter import FluxLocalImageInpainter

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
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

# Rango de tamaños (w, h) en px para imagen 1024x1024
# Calibrados para cámara cenital a 10-30m
OBJECT_SIZE_RANGES = {
    0: {"w": (30, 55),  "h": (15, 30)},   # plastic bottle: alargada
    1: {"w": (30, 55),  "h": (15, 30)},   # glass bottle: similar
    2: {"w": (20, 40),  "h": (20, 40)},   # can: cuadradita
    3: {"w": (50, 100), "h": (40, 90)},   # plastic bag: irregular, más grande
    4: {"w": (25, 60),  "h": (20, 50)},   # metal scrap: variable
    5: {"w": (40, 80),  "h": (30, 60)},   # plastic wrapper: rectangular
    6: {"w": (80, 160), "h": (60, 130)},  # trash pile: grande
    7: {"w": (20, 60),  "h": (20, 50)},   # trash: genérico
}

PROMPT_TEMPLATES = [
"{detail} floating on ocean surface, seen from elevated coastal viewpoint, "
    "natural lighting, photorealistic",

    "{detail} drifting on dark blue sea water, oblique aerial perspective, "
    "partially submerged, photorealistic, high resolution",
    
    "overhead drone view, {detail} drifting on sea water, "
    "partially submerged, natural lighting, photorealistic",

    "bird's eye view, {detail} on ocean surface with gentle ripples, "
    "realistic scale, high resolution drone photography",

    "zenithal drone capture, {detail} floating among small waves, "
    "midday sun, ultra realistic, ocean background",
]

CLASS_DETAILS = {
    0: ["crushed plastic water bottle", "deformed PET bottle", "floating plastic bottle"],
    1: ["green glass bottle", "clear glass bottle partially submerged", "beer glass bottle"],
    2: ["aluminum can", "crushed soda can", "rusted metal can"],
    3: ["white plastic bag spread on water", "translucent plastic bag partially underwater",
        "crumpled plastic bag floating"],
    4: ["rusty metal fragment", "small metal scrap piece", "bent metal debris"],
    5: ["candy wrapper floating", "crumpled plastic wrapper", "food packaging wrapper"],
    6: ["cluster of mixed floating debris", "small pile of trash on water surface",
        "accumulated floating garbage"],
    7: ["small piece of floating trash", "unidentified debris on water", "floating garbage"],
}

MIN_OBJECTS     = 1
MAX_OBJECTS     = 3
MIN_DIST_PX     = 100   # separación mínima entre objetos (Poisson disk)
MARGIN_PX       = 60    # margen desde el borde de imagen
FEATHER_RADIUS  = 10    # blur para suavizar máscara

INPUT_DIR       = "inputs"
OUTPUT_DIR      = "outputs"
IMG_SIZE        = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────

def is_water_region(image_np_rgb: np.ndarray, cx: int, cy: int,
                    check_radius: int = 50) -> bool:
    """
    Comprueba si la zona alrededor de (cx, cy) es agua.
    Criterios HSV más estrictos + verificación de uniformidad.
    """
    h, w = image_np_rgb.shape[:2]
    x0 = max(0, cx - check_radius)
    y0 = max(0, cy - check_radius)
    x1 = min(w, cx + check_radius)
    y1 = min(h, cy + check_radius)

    crop = image_np_rgb[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    avg_h = hsv[:, :, 0].mean()
    avg_s = hsv[:, :, 1].mean()
    avg_v = hsv[:, :, 2].mean()
    std_v = hsv[:, :, 2].std()

    # Agua oceánica: hue azulado (80-140), saturación moderada, brillo no extremo
    # std_v bajo = zona uniforme (agua), alto = textura compleja (edificio, costa)
    is_blue_hue = 80 <= avg_h <= 140
    has_saturation = avg_s > 20
    not_too_bright = avg_v < 170
    not_too_dark = avg_v > 30
    is_uniform = std_v < 45  # agua tiene textura suave

    return is_blue_hue and has_saturation and not_too_bright and not_too_dark and is_uniform


def poisson_disk_sampling(width: int, height: int, min_dist: int,
                          n_points: int, margin: int = 60,
                          max_attempts: int = 30) -> list:
    """
    Genera posiciones aleatorias con separación mínima garantizada.
    Mucho más natural que una cuadrícula.
    """
    points = []

    for _ in range(n_points * max_attempts):
        if len(points) >= n_points:
            break

        x = random.randint(margin, width - margin)
        y = random.randint(margin, height - margin)

        # Verificar distancia mínima con todos los puntos existentes
        too_close = False
        for px, py in points:
            if math.sqrt((x - px)**2 + (y - py)**2) < min_dist:
                too_close = True
                break

        if not too_close:
            points.append((x, y))

    return points


def get_object_size(class_id: int) -> tuple:
    """Tamaño aleatorio dentro del rango de la clase, con variación de aspecto."""
    ranges = OBJECT_SIZE_RANGES[class_id]
    w = random.randint(*ranges["w"])
    h = random.randint(*ranges["h"])
    # Pequeña rotación implícita: a veces intercambiar w/h
    if random.random() < 0.3:
        w, h = h, w
    return w, h


def make_feathered_mask(img_w: int, img_h: int,
                        cx: int, cy: int,
                        obj_w: int, obj_h: int,
                        blur_radius: int = 10) -> Image.Image:
    """
    Crea una máscara elíptica con bordes difuminados,
    centrada en (cx, cy) sobre un canvas del tamaño de la imagen.
    """
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    # Elipse con ligera perturbación angular
    angle_variation = random.uniform(-15, 15)  # no usamos rotación real,
    # pero variamos la relación de aspecto ligeramente
    aspect_noise = random.uniform(0.85, 1.15)
    ow = int(obj_w * aspect_noise)
    oh = int(obj_h / aspect_noise)

    draw.ellipse(
        [cx - ow // 2, cy - oh // 2, cx + ow // 2, cy + oh // 2],
        fill=255
    )

    # Suavizar bordes
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask


def build_prompt(class_id: int) -> str:
    """Prompt variado y específico por clase."""
    template = random.choice(PROMPT_TEMPLATES)
    detail = random.choice(CLASS_DETAILS[class_id])
    return template.format(detail=detail)


def compute_bbox_from_mask(mask: Image.Image, img_w: int, img_h: int) -> tuple:
    """
    Calcula el bounding box real a partir de la máscara (umbral > 127).
    Devuelve (x_center, y_center, w, h) normalizados.
    """
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 127)

    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_c = ((x_min + x_max) / 2) / img_w
    y_c = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    return x_c, y_c, w, h


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

print("Cargando FLUX Fill...")
inpainter = FluxLocalImageInpainter()

image_paths = (
    list(Path(INPUT_DIR).glob("*.jpg")) +
    list(Path(INPUT_DIR).glob("*.jpeg")) +
    list(Path(INPUT_DIR).glob("*.png"))
)
print(f"Imágenes encontradas: {len(image_paths)}")

for img_path in image_paths:
    print(f"\n{'='*60}")
    print(f"Procesando {img_path.name}...")
    image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image_np = np.array(image)

    # 1. Generar posiciones candidatas con Poisson disk
    n_objects = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    candidates = poisson_disk_sampling(
        IMG_SIZE, IMG_SIZE,
        min_dist=MIN_DIST_PX,
        n_points=n_objects * 3,  # generar más de los necesarios
        margin=MARGIN_PX,
    )

    # 2. Filtrar: solo posiciones que sean agua
    water_positions = [
        (cx, cy) for cx, cy in candidates
        if is_water_region(image_np, cx, cy, check_radius=50)
    ]
    print(f"  Candidatos: {len(candidates)} | Agua válida: {len(water_positions)}")

    if not water_positions:
        print(f"  ⚠️  Sin posiciones válidas de agua, saltando.")
        continue

    # Tomar hasta n_objects posiciones
    chosen_positions = water_positions[:n_objects]
    print(f"  Objetos a insertar: {len(chosen_positions)}")

    # 3. Insertar objetos uno a uno
    annotations = []
    for cx, cy in chosen_positions:
        class_id = random.choice(list(CLASSES.keys()))
        class_name = CLASSES[class_id]
        obj_w, obj_h = get_object_size(class_id)
        prompt = build_prompt(class_id)

        print(f"  → '{class_name}' en ({cx},{cy}), tamaño {obj_w}x{obj_h}px")
        print(f"    Prompt: {prompt[:80]}...")

        # Crear máscara feathered
        mask = make_feathered_mask(
            IMG_SIZE, IMG_SIZE, cx, cy,
            obj_w, obj_h,
            blur_radius=FEATHER_RADIUS
        )

        # Inpaint
        image = inpainter.inpaint(image, mask, prompt)

        # Calcular bbox desde la máscara real
        bbox = compute_bbox_from_mask(mask, IMG_SIZE, IMG_SIZE)
        if bbox:
            x_c, y_c, w, h = bbox
            annotations.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # 4. Guardar
    stem = img_path.stem
    out_img = f"{OUTPUT_DIR}/{stem}_synth.png"
    out_txt = f"{OUTPUT_DIR}/{stem}_synth.txt"
    image.save(out_img)
    with open(out_txt, "w") as f:
        f.write("\n".join(annotations))
    print(f"  ✓ Guardado: {out_img} ({len(annotations)} anotaciones)")

print(f"\n{'='*60}")
print("Generación completada.")
