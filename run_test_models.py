"""
Alpha5 — Script de prueba comparativa de modelos
Genera una imagen de test con cada modelo (Canny, Redux, Kontext)
para evaluar visualmente cuál integra mejor los objetos.

Uso:
    python run_test_models.py --model canny
    python run_test_models.py --model redux
    python run_test_models.py --model kontext
    python run_test_models.py --model all      # Prueba los tres (lento)
    python run_test_models.py                  # Por defecto: 'all'

Genera en outputs_test/{model}/{stem}_result.png y _debug.png.
"""

import argparse
import csv
import math
import os
import random

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

INPUT_DIR    = "inputs"
OUTPUT_DIR   = "outputs_test"
PROMPTS_CSV  = "config/prompts.csv"

MAX_SIDE     = 1024
DIVISOR      = 16
EDGE_MARGIN  = 80
WATER_CHECK_MARGIN = 30

# Un solo objeto por imagen en el test para comparar limpiamente
TEST_CLASS_ID = 0   # plastic bottle (cambiar para probar otras clases)

OBJECT_SIZES = {
    0: (100, 180,  50,  90),
    1: (100, 180,  50,  90),
    2: ( 80, 140,  70, 130),
    3: (150, 280, 120, 240),
    4: (100, 200,  80, 160),
    5: (120, 220,  80, 160),
    6: (220, 380, 180, 320),
    7: ( 90, 180,  70, 150),
}

MODELS = ["canny", "redux", "kontext"]

LOG_FIELDS = [
    "image_out", "source_image",
    "class_id", "class_name", "prompt", "model",
    "cx", "cy", "obj_w", "obj_h",
    "bbox_xc", "bbox_yc", "bbox_w", "bbox_h",
]


# ═══════════════════════════════════════════════════════════════
# HELPERS (reutilizados de run_fill.py)
# ═══════════════════════════════════════════════════════════════

def load_class_names(csv_path: str) -> dict:
    class_names = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            if cid not in class_names:
                class_names[cid] = row["class_name"].strip()
    return class_names


def load_prompts(csv_path: str) -> dict:
    prompts_by_class = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            prompt = row["prompt"].strip().strip('"')
            prompts_by_class.setdefault(cid, []).append(prompt)
    return prompts_by_class


def prepare_image(image: Image.Image, max_side: int = 1024, divisor: int = 16):
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(divisor, round(w * scale / divisor) * divisor)
    new_h = max(divisor, round(h * scale / divisor) * divisor)
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def is_water_region(image_np, cx, cy, half_w, half_h, margin=30):
    img_h, img_w = image_np.shape[:2]
    x0, y0 = cx - half_w - margin, cy - half_h - margin
    x1, y1 = cx + half_w + margin, cy + half_h + margin
    if x0 < 0 or y0 < 0 or x1 >= img_w or y1 >= img_h:
        return False
    crop = image_np[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    avg_h, avg_s, avg_v = hsv[:,:,0].mean(), hsv[:,:,1].mean(), hsv[:,:,2].mean()
    std_v = hsv[:,:,2].std()
    edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), 50, 150)
    edge_density = edges.mean() / 255.0
    return all([
        70 <= avg_h <= 145,
        avg_s > 15,
        30 < avg_v < 210,
        std_v < 45,
        edge_density < 0.08,
    ])


def find_water_position(image_np, class_id, margin=80, max_tries=200):
    img_h, img_w = image_np.shape[:2]
    min_w, max_w, min_h, max_h = OBJECT_SIZES[class_id]
    for _ in range(max_tries):
        cx = random.randint(margin, img_w - margin)
        cy = random.randint(margin, img_h - margin)
        obj_w = random.randint(min_w, max_w)
        obj_h = random.randint(min_h, max_h)
        if is_water_region(image_np, cx, cy, obj_w // 2, obj_h // 2, WATER_CHECK_MARGIN):
            return cx, cy, obj_w, obj_h
    return None


def create_mask(img_w, img_h, cx, cy, obj_w, obj_h, blur_radius=4):
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        [cx - obj_w // 2, cy - obj_h // 2, cx + obj_w // 2, cy + obj_h // 2],
        fill=255,
    )
    return mask.filter(ImageFilter.GaussianBlur(radius=blur_radius)) if blur_radius > 0 else mask


def save_debug_image(image, annotations, path):
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
# CARGA LAZY DE MODELOS
# ═══════════════════════════════════════════════════════════════

def load_model(model_name: str):
    if model_name == "canny":
        from core.dependencies.ai.generative_ai.image_inpainters.flux_canny_inpainter import (
            FluxCannyInpainter,
        )
        return FluxCannyInpainter()

    elif model_name == "redux":
        from core.dependencies.ai.generative_ai.image_inpainters.flux_redux_inpainter import (
            FluxReduxInpainter,
        )
        return FluxReduxInpainter()

    elif model_name == "kontext":
        from core.dependencies.ai.generative_ai.image_inpainters.flux_kontext_inpainter import (
            FluxKontextInpainter,
        )
        return FluxKontextInpainter()

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


# ═══════════════════════════════════════════════════════════════
# LÓGICA DE TEST POR MODELO
# ═══════════════════════════════════════════════════════════════

def run_inpaint(model_name, model, image, mask, prompt, class_id):
    """Interfaz unificada que maneja las particularidades de cada modelo."""
    if model_name == "redux":
        return model.inpaint(image, mask, prompt, class_id=class_id), None
    elif model_name == "kontext":
        result = model.inpaint(image, mask, prompt)
        bbox = model.compute_bbox(image, result)
        return result, bbox
    else:
        return model.inpaint(image, mask, prompt), None


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def test_model(model_name: str, image_paths: list, prompts_by_class: dict, class_names: dict):
    out_dir = Path(OUTPUT_DIR) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "generation_log.csv"
    log_exists = log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if not log_exists:
        log_writer.writeheader()

    print(f"\n{'═' * 60}")
    print(f"Cargando modelo: {model_name.upper()}")
    model = load_model(model_name)
    print(f"  ✓ Listo")

    for img_path in image_paths:
        print(f"\n  [{img_path.name}]")
        image = Image.open(img_path).convert("RGB")
        image, scale = prepare_image(image, MAX_SIDE, DIVISOR)
        img_w, img_h = image.size
        image_np = np.array(image)

        pos = find_water_position(image_np, TEST_CLASS_ID)
        if pos is None:
            print(f"    ⚠️  No se encontró zona de agua, saltando.")
            continue

        cx, cy, obj_w, obj_h = pos
        prompt = random.choice(prompts_by_class[TEST_CLASS_ID])
        mask = create_mask(img_w, img_h, cx, cy, obj_w, obj_h)

        print(f"    Clase={TEST_CLASS_ID} en ({cx},{cy}) tamaño={obj_w}×{obj_h}px")
        print(f"    Prompt: {prompt[:60]}...")

        result, external_bbox = run_inpaint(model_name, model, image, mask, prompt, TEST_CLASS_ID)

        # Bbox: prioridad al bbox externo (Kontext diff), si no, del mask
        if external_bbox:
            xc, yc, bw, bh = external_bbox
        else:
            mask_np = np.array(mask)
            ys, xs = np.where(mask_np > 127)
            xc = ((xs.min() + xs.max()) / 2.0) / img_w
            yc = ((ys.min() + ys.max()) / 2.0) / img_h
            bw = (xs.max() - xs.min()) / img_w
            bh = (ys.max() - ys.min()) / img_h

        ann = f"{TEST_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        stem = img_path.stem

        result.save(out_dir / f"{stem}_result.png")
        save_debug_image(result, [ann], out_dir / f"{stem}_debug.png")
        with open(out_dir / f"{stem}.txt", "w") as f:
            f.write(ann)

        log_writer.writerow({
            "image_out":    str(out_dir / f"{stem}_result.png"),
            "source_image": img_path.name,
            "class_id":     TEST_CLASS_ID,
            "class_name":   class_names.get(TEST_CLASS_ID, ""),
            "prompt":       prompt,
            "model":        model_name,
            "cx":           cx,
            "cy":           cy,
            "obj_w":        obj_w,
            "obj_h":        obj_h,
            "bbox_xc":      f"{xc:.6f}",
            "bbox_yc":      f"{yc:.6f}",
            "bbox_w":       f"{bw:.6f}",
            "bbox_h":       f"{bh:.6f}",
        })

        print(f"    ✓ Guardado en {out_dir / stem}_result.png")

    log_file.close()
    print(f"\n  ✓ Modelo {model_name} completado. Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Test comparativo de modelos FLUX")
    parser.add_argument(
        "--model",
        choices=MODELS + ["all"],
        default="all",
        help="Modelo a probar (canny, redux, kontext, all)",
    )
    parser.add_argument(
        "--input",
        default=INPUT_DIR,
        help="Directorio de imágenes de entrada",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Número máximo de imágenes a procesar por modelo",
    )
    args = parser.parse_args()

    prompts_by_class = load_prompts(PROMPTS_CSV)
    class_names = load_class_names(PROMPTS_CSV)

    image_paths = sorted(
        p for p in Path(args.input).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[: args.max_images]

    if not image_paths:
        print(f"No se encontraron imágenes en {args.input}")
        return

    print(f"Imágenes: {[p.name for p in image_paths]}")

    models_to_test = MODELS if args.model == "all" else [args.model]

    for model_name in models_to_test:
        test_model(model_name, image_paths, prompts_by_class, class_names)

    print(f"\n{'═' * 60}")
    print(f"Tests completados. Resultados en {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
