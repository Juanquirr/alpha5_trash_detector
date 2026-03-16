import argparse
import csv
import os
import random
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import cv2

from core.dependencies.ai.generative_ai.image_inpainters.flux_local_image_inpainter import (
    FluxLocalImageInpainter,
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

INPUT_DIR   = "inputs"
OUTPUT_DIR  = "outputs"
PROMPTS_CSV = "config/prompts.csv"
LOG_CSV     = f"{OUTPUT_DIR}/generation_log.csv"

LOG_FIELDS  = [
    "image_out", "source_image",
    "class_id", "class_name", "prompt",
    "cx", "cy", "obj_w", "obj_h",
    "bbox_xc", "bbox_yc", "bbox_w", "bbox_h",
]

MAX_SIDE    = 1024      # Lado mayor tras resize (mantiene aspect ratio)
DIVISOR     = 16        # FLUX requiere dimensiones múltiplo de 16

MIN_OBJECTS = 2
MAX_OBJECTS = 3
MIN_DIST_PX = 180       # Separación mínima entre objetos (Poisson disk)
EDGE_MARGIN = 80        # Margen mínimo desde el borde de la imagen
WATER_CHECK_MARGIN = 30 # Margen extra para verificar que toda la zona es agua

# Tamaños en píxeles (para imagen con lado mayor ~1024px)
# (min_w, max_w, min_h, max_h) — lo suficientemente grandes para ser detectables
OBJECT_SIZES = {
    0: (100, 180,  50,  90),   # plastic bottle — alargada horizontal
    1: (100, 180,  50,  90),   # glass bottle   — similar
    2: ( 80, 140,  70, 130),   # can            — más cuadrada
    3: (150, 280, 120, 240),   # plastic bag    — grande, se extiende en agua
    4: (100, 200,  80, 160),   # metal scrap    — irregular
    5: (120, 220,  80, 160),   # plastic wrapper — rectangular
    6: (220, 380, 180, 320),   # trash pile     — grande, cluster
    7: ( 90, 180,  70, 150),   # trash          — genérico
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CARGA DE PROMPTS
# ═══════════════════════════════════════════════════════════════

def load_prompts(csv_path: str) -> dict:
    """
    Carga prompts.csv y devuelve un dict {class_id: [lista de prompts]}.
    """
    prompts_by_class = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            prompt = row["prompt"].strip().strip('"')
            if cid not in prompts_by_class:
                prompts_by_class[cid] = []
            prompts_by_class[cid].append(prompt)
    return prompts_by_class


def load_class_names(csv_path: str) -> dict:
    """
    Carga prompts.csv y devuelve un dict {class_id: class_name}.
    """
    class_names = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            if cid not in class_names:
                class_names[cid] = row["class_name"].strip()
    return class_names


# ═══════════════════════════════════════════════════════════════
# PREPARACIÓN DE IMAGEN (aspect ratio preservado)
# ═══════════════════════════════════════════════════════════════

def prepare_image(image: Image.Image, max_side: int = 1024, divisor: int = 16):
    """
    Redimensiona manteniendo aspect ratio. Redondea a múltiplos de `divisor`.
    Devuelve (imagen_redimensionada, escala_aplicada).
    """
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)  # no agrandar
    new_w = max(divisor, round(w * scale / divisor) * divisor)
    new_h = max(divisor, round(h * scale / divisor) * divisor)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    return resized, scale


# ═══════════════════════════════════════════════════════════════
# DETECCIÓN DE AGUA (filtro robusto)
# ═══════════════════════════════════════════════════════════════

def is_water_region(image_np: np.ndarray, cx: int, cy: int,
                    half_w: int, half_h: int, margin: int = 30) -> bool:
    """
    Verifica que TODA la zona (objeto + margen) sea agua.
    Usa: hue azul-verde, saturación, brillo moderado, textura suave, pocas aristas.
    """
    img_h, img_w = image_np.shape[:2]

    # Región a comprobar = tamaño del objeto + margen de seguridad
    x0 = cx - half_w - margin
    y0 = cy - half_h - margin
    x1 = cx + half_w + margin
    y1 = cy + half_h + margin

    # ¿Cabe en la imagen?
    if x0 < 0 or y0 < 0 or x1 >= img_w or y1 >= img_h:
        return False

    crop = image_np[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    avg_h = hsv[:, :, 0].mean()   # Hue (0-180 en OpenCV)
    avg_s = hsv[:, :, 1].mean()   # Saturación
    avg_v = hsv[:, :, 2].mean()   # Brillo
    std_v = hsv[:, :, 2].std()    # Uniformidad de textura

    # Densidad de aristas — alto = estructuras, bajo = agua
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0

    # Criterios: agua oceánica/costera
    is_water_hue     = 70 <= avg_h <= 145     # Azul-verde-teal
    has_saturation   = avg_s > 15
    moderate_bright  = 30 < avg_v < 210
    is_smooth        = std_v < 45             # Agua = textura suave
    few_edges        = edge_density < 0.08    # Sin estructuras

    return all([is_water_hue, has_saturation, moderate_bright, is_smooth, few_edges])


# ═══════════════════════════════════════════════════════════════
# DISTRIBUCIÓN ESPACIAL (Poisson Disk)
# ═══════════════════════════════════════════════════════════════

def poisson_disk_sampling(
    width: int, height: int,
    min_dist: int, n_points: int,
    margin: int = 80, max_attempts: int = 200,
) -> list:
    """
    Genera posiciones aleatorias con separación mínima garantizada.
    Más natural que una cuadrícula.
    """
    points = []
    for _ in range(n_points * max_attempts):
        if len(points) >= n_points:
            break
        x = random.randint(margin, width - margin)
        y = random.randint(margin, height - margin)
        if all(math.hypot(x - px, y - py) >= min_dist for px, py in points):
            points.append((x, y))
    return points


# ═══════════════════════════════════════════════════════════════
# TAMAÑO Y MÁSCARA DEL OBJETO
# ═══════════════════════════════════════════════════════════════

def get_object_size(class_id: int) -> tuple:
    """Tamaño aleatorio dentro del rango de la clase."""
    min_w, max_w, min_h, max_h = OBJECT_SIZES[class_id]
    w = random.randint(min_w, max_w)
    h = random.randint(min_h, max_h)
    # Ocasionalmente intercambiar ejes (rotación 90°)
    if random.random() < 0.3:
        w, h = h, w
    return w, h


def create_mask(img_w: int, img_h: int,
                cx: int, cy: int,
                obj_w: int, obj_h: int,
                blur_radius: int = 5) -> Image.Image:
    """
    Máscara elíptica binaria con ligero suavizado de bordes.
    FLUX Fill espera máscara casi binaria; un blur leve ayuda en el blend.
    """
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        [cx - obj_w // 2, cy - obj_h // 2,
         cx + obj_w // 2, cy + obj_h // 2],
        fill=255,
    )
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask


# ═══════════════════════════════════════════════════════════════
# BOUNDING BOX DESDE MÁSCARA → YOLO
# ═══════════════════════════════════════════════════════════════

def compute_yolo_bbox(mask: Image.Image) -> tuple:
    """
    Calcula bbox YOLO normalizado desde la máscara (umbral > 127).
    Retorna (x_center, y_center, width, height) normalizados, o None.
    """
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        return None

    img_w, img_h = mask.size  # PIL: (w, h)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_c = ((x_min + x_max) / 2.0) / img_w
    y_c = ((y_min + y_max) / 2.0) / img_h
    w   = (x_max - x_min) / img_w
    h   = (y_max - y_min) / img_h

    return x_c, y_c, w, h


# ═══════════════════════════════════════════════════════════════
# VISUALIZACIÓN DE DEBUG (opcional)
# ═══════════════════════════════════════════════════════════════

def save_debug_image(image: Image.Image, annotations: list, path: str):
    """Guarda copia con bounding boxes dibujados para inspección visual."""
    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    iw, ih = debug.size
    colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "white"]

    for ann in annotations:
        parts = ann.split()
        cid = int(parts[0])
        xc, yc, w, h = [float(x) for x in parts[1:]]
        x0 = int((xc - w / 2) * iw)
        y0 = int((yc - h / 2) * ih)
        x1 = int((xc + w / 2) * iw)
        y1 = int((yc + h / 2) * ih)
        color = colors[cid % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"cls={cid}", fill=color)
    debug.save(path)


# ═══════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generador de imágenes sintéticas con FLUX Fill")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Número exacto de instancias por imagen (por defecto: aleatorio 2-3)",
    )
    args = parser.parse_args()

    # 1. Cargar prompts y nombres de clase
    print("Cargando prompts desde CSV...")
    prompts_by_class = load_prompts(PROMPTS_CSV)
    class_names = load_class_names(PROMPTS_CSV)
    class_ids = list(prompts_by_class.keys())
    print(f"  Clases disponibles: {len(class_ids)}")

    # 2. Cargar modelo
    print("Cargando FLUX Fill (puede tardar ~30s)...")
    inpainter = FluxLocalImageInpainter()
    print("  ✓ Modelo cargado")

    # 3. Abrir log CSV (append; cabecera solo si es nuevo)
    log_exists = Path(LOG_CSV).exists()
    log_file = open(LOG_CSV, "a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if not log_exists:
        log_writer.writeheader()

    # 4. Listar imágenes de entrada
    image_paths = sorted(
        p for p in Path(INPUT_DIR).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Imágenes de entrada: {len(image_paths)}")

    # 5. Procesar cada imagen
    for img_idx, img_path in enumerate(image_paths):
        print(f"\n{'═' * 60}")
        print(f"[{img_idx + 1}/{len(image_paths)}] {img_path.name}")

        image = Image.open(img_path).convert("RGB")
        print(f"  Original: {image.size[0]}×{image.size[1]}")

        # 4a. Redimensionar conservando aspect ratio
        image, scale = prepare_image(image, max_side=MAX_SIDE, divisor=DIVISOR)
        img_w, img_h = image.size
        print(f"  Redimensionada: {img_w}×{img_h} (scale={scale:.3f})")

        image_np = np.array(image)

        # 4b. Generar posiciones candidatas
        n_objects = args.num_instances if args.num_instances is not None else random.randint(MIN_OBJECTS, MAX_OBJECTS)
        candidates = poisson_disk_sampling(
            img_w, img_h,
            min_dist=MIN_DIST_PX,
            n_points=n_objects * 4,  # generar de más, luego filtrar
            margin=EDGE_MARGIN,
        )
        print(f"  Candidatos generados: {len(candidates)}")

        # 4c. Filtrar: solo posiciones sobre agua
        valid_positions = []
        for cx, cy in candidates:
            # Pre-calcular tamaño para verificar que quepa
            test_class = random.choice(class_ids)
            test_w, test_h = get_object_size(test_class)
            if is_water_region(image_np, cx, cy, test_w // 2, test_h // 2,
                               margin=WATER_CHECK_MARGIN):
                valid_positions.append((cx, cy))

            if len(valid_positions) >= n_objects:
                break

        print(f"  Posiciones válidas (agua): {len(valid_positions)}")

        if not valid_positions:
            print(f"  ⚠️  Sin posiciones de agua válidas, saltando imagen.")
            continue

        # 5d. Insertar objetos
        annotations = []
        for pos_idx, (cx, cy) in enumerate(valid_positions):
            class_id = random.choice(class_ids)
            prompt = random.choice(prompts_by_class[class_id])
            obj_w, obj_h = get_object_size(class_id)

            print(f"  [{pos_idx + 1}/{len(valid_positions)}] "
                  f"clase={class_id} ({OBJECT_SIZES[class_id][:2][0]}-{OBJECT_SIZES[class_id][1]}px) "
                  f"en ({cx},{cy}), tamaño={obj_w}×{obj_h}px")

            # Crear máscara
            mask = create_mask(img_w, img_h, cx, cy, obj_w, obj_h, blur_radius=4)

            # Inpaint
            image = inpainter.inpaint(image, mask, prompt)

            # Actualizar numpy para siguientes verificaciones de agua
            image_np = np.array(image)

            # Calcular bbox YOLO
            bbox = compute_yolo_bbox(mask)
            if bbox:
                xc, yc, w, h = bbox
                annotations.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                print(f"    → bbox: center=({xc:.3f},{yc:.3f}) size=({w:.3f},{h:.3f})")
                log_writer.writerow({
                    "image_out":   f"{OUTPUT_DIR}/{img_path.stem}_synth.png",
                    "source_image": img_path.name,
                    "class_id":    class_id,
                    "class_name":  class_names.get(class_id, ""),
                    "prompt":      prompt,
                    "cx":          cx,
                    "cy":          cy,
                    "obj_w":       obj_w,
                    "obj_h":       obj_h,
                    "bbox_xc":     f"{xc:.6f}",
                    "bbox_yc":     f"{yc:.6f}",
                    "bbox_w":      f"{w:.6f}",
                    "bbox_h":      f"{h:.6f}",
                })

        # 5e. Guardar resultados
        stem = img_path.stem
        out_img_path = f"{OUTPUT_DIR}/{stem}_synth.png"
        out_txt_path = f"{OUTPUT_DIR}/{stem}_synth.txt"
        out_dbg_path = f"{OUTPUT_DIR}/{stem}_debug.png"

        image.save(out_img_path)
        with open(out_txt_path, "w") as f:
            f.write("\n".join(annotations))

        # Debug: imagen con bboxes dibujados
        save_debug_image(image, annotations, out_dbg_path)

        print(f"  ✓ {out_img_path} ({len(annotations)} objetos)")
        print(f"  ✓ {out_dbg_path} (visualización de debug)")

    log_file.close()
    print(f"\n{'═' * 60}")
    print(f"Generación completada. Log: {LOG_CSV}")


if __name__ == "__main__":
    main()