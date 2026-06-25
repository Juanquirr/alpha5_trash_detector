"""
grounding_eval.py  --  VLM visual grounding evaluation.

Ask a Qwen VLM to locate trash objects with bounding boxes, then compare
against YOLO ground-truth annotations via IoU.

Pipeline:
  1. Send image + structured prompt asking for class + bbox per object.
  2. Parse coordinates from the (often inconsistent) free-text response.
  3. Match predicted boxes to YOLO GT boxes using greedy IoU matching.
  4. Report precision, recall, and mean IoU at configurable threshold.

Supported models: qwen_vl (Qwen2.5-VL-3B), qwen_2b (Qwen3-VL-2B).
Other models do not support structured bounding box output.

Usage:
    python grounding_eval.py --model qwen_vl --dataset ../alpha5/datasets/alpha7
    python grounding_eval.py --model qwen_2b --dataset ../alpha5/datasets/alpha7 --limit 50
    python grounding_eval.py --model qwen_vl --images imgs/ --labels lbls/ --limit 100
    python grounding_eval.py --model qwen_2b --dataset ../alpha5/datasets/alpha7 --lora-adapter pope_results_v7/qwen_2b_lora
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

DEFAULT_CLASSES = [
    "container", "plastic", "metal", "polystyrene",
    "trash_pile", "trash",
]

YOLO_ID_TO_CLASS = {i: c for i, c in enumerate(DEFAULT_CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

GROUNDING_PROMPT = (
    "Detect all waste objects in this image. For each object, output its class "
    "and bounding box.\n\n"
    "Classes:\n"
    "- container: rigid non-metal container — bottles, jars, rigid cups\n"
    "- plastic: flat flexible plastic — bags, film, soft wrappers\n"
    "- metal: metallic reflection — cans, aluminium foil, metal scrap\n"
    "- polystyrene: white matte foam — EPS blocks, foam cups\n"
    "- trash pile: dense cluster of mixed waste\n"
    "- trash: unclassifiable single waste item\n\n"
    "Format each detection as:\n"
    "<ref>class_name</ref><box>(x1, y1, x2, y2)</box>\n\n"
    "Coordinates are pixel positions. Output one line per object.\n"
    "If no waste is visible, output only: CLEAN"
)

CSV_FIELDS = [
    "image", "n_gt", "n_pred", "n_matched",
    "precision", "recall", "mean_iou",
    "gt_classes", "pred_classes", "pred_boxes",
    "inference_s",
]


# -- Visualization ----------------------------------------------------------

def draw_boxes(image, gt_boxes, pred_boxes, matches, out_path):
    """Save annotated image: GT=green, matched pred=cyan, unmatched pred=red."""
    from PIL import ImageDraw, ImageFont
    img = image.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    def px(box):
        return [box[0] * W, box[1] * H, box[2] * W, box[3] * H]

    matched_pred_indices = set()
    for m in matches:
        if m["matched"]:
            for j, pred in enumerate(pred_boxes):
                if pred["cls"] == m["pred_cls"] and j not in matched_pred_indices:
                    matched_pred_indices.add(j)
                    break

    for b in gt_boxes:
        x1, y1, x2, y2 = px(b["box"])
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1 + 2, y1 + 2), f"GT:{b['cls']}", fill="lime")

    for j, pred in enumerate(pred_boxes):
        x1, y1, x2, y2 = px(pred["box"])
        color = "cyan" if j in matched_pred_indices else "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, y2 - 14), pred["cls"], fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# -- Bounding box parsing ---------------------------------------------------

# 4 coordinates inside parentheses; each coord may have a leading letter
# prefix (x0, y153, b42) which the model sometimes emits.
_COORDS_RE = re.compile(
    r"\(\s*[a-zA-Z]?\s*(\d+\.?\d*)\s*,\s*[a-zA-Z]?\s*(\d+\.?\d*)"
    r"\s*,\s*[a-zA-Z]?\s*(\d+\.?\d*)\s*,\s*[a-zA-Z]?\s*(\d+\.?\d*)\s*\)"
)


def _normalize_coords(coords: list[float], img_w: int, img_h: int) -> list[float]:
    """Normalize (x1,y1,x2,y2) to [0,1]. Pixel coords divided by image size."""
    if max(coords) <= 1.0:
        return coords
    return [
        coords[0] / img_w, coords[1] / img_h,
        coords[2] / img_w, coords[3] / img_h,
    ]


def _extract_class_from_text(text: str) -> str | None:
    """Clean a text fragment and try to extract a class name from it."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[\s:,>]+$", "", text).strip()
    if not text:
        return None
    raw = text.lower().replace(" ", "_").replace("-", "_")
    return _match_class(raw)


def parse_grounding_response(response: str, img_w: int, img_h: int) -> list[dict]:
    """
    Parse Qwen grounding output into list of {cls, box}.
    box = (x1, y1, x2, y2) normalized to [0, 1].

    Uses finditer over the full response to handle:
      - Multiple detections on a single line
      - Class name on a separate line from coordinates
      - Malformed / missing XML tags
      - Letter-prefixed coordinates (x0, y153, b42)
    """
    detections = []
    seen: set[tuple] = set()
    prev_end = 0
    last_cls = None

    for m in _COORDS_RE.finditer(response):
        coords = [float(m.group(i)) for i in range(1, 5)]

        between = response[prev_end:m.start()]
        prev_end = m.end()

        cls = _extract_class_from_text(between)
        if cls:
            last_cls = cls
        elif last_cls:
            cls = last_cls
        else:
            continue

        norm = _normalize_coords(coords, img_w, img_h)
        norm = [round(max(0.0, min(1.0, c)), 4) for c in norm]

        key = (cls, norm[0], norm[1], norm[2], norm[3])
        if key not in seen:
            seen.add(key)
            detections.append({"cls": cls, "box": tuple(norm)})

    return detections


def _match_class(raw: str, classes: list[str] | None = None) -> str | None:
    """Fuzzy-match a raw class string to known classes."""
    if classes is None:
        classes = DEFAULT_CLASSES
    norm = raw.strip().lower().replace(" ", "_")
    if norm in classes:
        return norm
    for cls in sorted(classes, key=len, reverse=True):
        if cls in norm or norm in cls:
            return cls
    return None


# -- IoU computation --------------------------------------------------------

def iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) normalized boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def match_predictions(gt_boxes: list[dict], pred_boxes: list[dict],
                      iou_threshold: float = 0.3) -> list[dict]:
    """
    Greedy matching: each GT box matched to best-IoU prediction (same class).
    Returns list of {gt_cls, pred_cls, iou, matched}.
    """
    matches   = []
    used_pred = set()

    for gt in gt_boxes:
        best_iou  = 0.0
        best_idx  = -1

        for j, pred in enumerate(pred_boxes):
            if j in used_pred:
                continue
            if pred["cls"] != gt["cls"]:
                continue
            score = iou(gt["box"], pred["box"])
            if score > best_iou:
                best_iou = score
                best_idx = j

        if best_idx >= 0 and best_iou >= iou_threshold:
            used_pred.add(best_idx)
            matches.append({
                "gt_cls": gt["cls"], "pred_cls": pred_boxes[best_idx]["cls"],
                "iou": round(best_iou, 3), "matched": True,
            })
        else:
            matches.append({
                "gt_cls": gt["cls"], "pred_cls": None,
                "iou": round(best_iou, 3) if best_idx >= 0 else 0.0,
                "matched": False,
            })

    return matches


# -- YOLO label loading -----------------------------------------------------

def load_yolo_gt(label_path: Path) -> list[dict]:
    """Load YOLO .txt label -> list of {cls, box} with normalized (x1,y1,x2,y2)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except (ValueError, IndexError):
            continue
        if cid not in YOLO_ID_TO_CLASS:
            continue
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes.append({"cls": YOLO_ID_TO_CLASS[cid], "box": (x1, y1, x2, y2)})
    return boxes


# -- Image collection -------------------------------------------------------

def collect_images(dataset: Path | None, images_dir: Path | None,
                   labels_dir: Path | None, limit: int) -> list[dict]:
    """
    Collect (image_path, label_path) pairs.
    Returns list of {image: Path, label: Path}.
    """
    pairs = []

    # Skip preview/debug images that have boxes or masks drawn on top, so the
    # VLM only sees the clean scene (never an image with the GT box overlaid).
    def _is_debug(img):
        return "_debug" in img.stem

    if dataset:
        for split in ("val", "train", "test"):
            img_dir = dataset / split / "images"
            lbl_dir = dataset / split / "labels"
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTS and not _is_debug(img):
                    lbl = lbl_dir / (img.stem + ".txt")
                    pairs.append({"image": img, "label": lbl})
    else:
        img_dir = images_dir or Path("images")
        lbl_dir = labels_dir or img_dir
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS and not _is_debug(img):
                lbl = lbl_dir / (img.stem + ".txt")
                pairs.append({"image": img, "label": lbl})

    with_labels    = [p for p in pairs if p["label"].exists() and p["label"].stat().st_size > 0]
    without_labels = [p for p in pairs if p not in with_labels]

    selected = with_labels[:limit]
    remaining = limit - len(selected)
    if remaining > 0:
        selected += without_labels[:remaining]

    return selected


# -- Model inference --------------------------------------------------------

def run_grounding(model_key: str, pairs: list[dict], out_csv: Path,
                  iou_threshold: float, lora_adapter: Path | None = None,
                  viz_dir: Path | None = None) -> None:
    """Run grounding inference and save results."""
    import torch
    from PIL import Image
    from models import REGISTRY

    vlm_cls = REGISTRY[model_key]
    vlm     = vlm_cls()
    print(f"\n[{model_key}] Loading model...")
    vlm.load()

    if lora_adapter is not None:
        from peft import PeftModel
        print(f"[{model_key}] Loading LoRA adapter from {lora_adapter}...")
        vlm.model = PeftModel.from_pretrained(vlm.model, str(lora_adapter))
        vlm.model = vlm.model.merge_and_unload()
        print(f"[{model_key}] LoRA adapter merged.")

    is_cuda = vlm.device == "cuda" and torch.cuda.is_available()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()

    total_gt      = 0
    total_pred    = 0
    total_matched = 0
    all_ious      = []

    print(f"[{model_key}] Processing {len(pairs)} images...\n")
    t_start = time.perf_counter()

    for i, pair in enumerate(pairs, 1):
        image = Image.open(pair["image"]).convert("RGB")
        img_w, img_h = image.size

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": GROUNDING_PROMPT},
                ],
            }
        ]
        text = vlm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = vlm.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(vlm.device)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = vlm.model.generate(**inputs, max_new_tokens=500)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = round(time.perf_counter() - t0, 3)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        response  = vlm.processor.decode(generated[0], skip_special_tokens=True).strip()

        preds   = parse_grounding_response(response, img_w, img_h)
        gt      = load_yolo_gt(pair["label"])
        matches = match_predictions(gt, preds, iou_threshold)

        n_matched  = sum(1 for m in matches if m["matched"])
        match_ious = [m["iou"] for m in matches if m["matched"]]

        total_gt      += len(gt)
        total_pred    += len(preds)
        total_matched += n_matched
        all_ious.extend(match_ious)

        prec = n_matched / len(preds) * 100 if preds else float("nan")
        rec  = n_matched / len(gt) * 100 if gt else float("nan")
        miou = sum(match_ious) / len(match_ious) if match_ious else 0.0

        pred_boxes_json = json.dumps(
            [{"cls": p["cls"], "box": list(p["box"])} for p in preds],
            ensure_ascii=False,
        )

        row = {
            "image":       pair["image"].name,
            "n_gt":        len(gt),
            "n_pred":      len(preds),
            "n_matched":   n_matched,
            "precision":   round(prec, 1),
            "recall":      round(rec, 1),
            "mean_iou":    round(miou, 3),
            "gt_classes":  ", ".join(sorted({b["cls"] for b in gt})),
            "pred_classes": ", ".join(sorted({b["cls"] for b in preds})),
            "pred_boxes":  pred_boxes_json,
            "inference_s": elapsed,
        }

        with out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)

        if viz_dir is not None:
            draw_boxes(image, gt, preds, matches,
                       viz_dir / f"{pair['image'].stem}_viz.jpg")

        ok = "+" if n_matched > 0 else "-"
        eta_s = (time.perf_counter() - t_start) / i * (len(pairs) - i)
        eta_m = int(eta_s // 60)
        print(
            f"  [{i}/{len(pairs)}] {ok} {pair['image'].name[:25]:<25s}  "
            f"GT={len(gt)} Pred={len(preds)} Match={n_matched}  "
            f"IoU={miou:.2f}  | {elapsed}s  | ETA {eta_m}m"
        )

        if is_cuda and i % 50 == 0:
            torch.cuda.empty_cache()

    vlm.unload()

    # -- Summary -------------------------------------------------------------
    total_time = time.perf_counter() - t_start
    overall_prec = total_matched / total_pred * 100 if total_pred > 0 else 0
    overall_rec  = total_matched / total_gt * 100 if total_gt > 0 else 0
    overall_iou  = sum(all_ious) / len(all_ious) if all_ious else 0

    print(f"\n{'='*60}")
    print(f"  GROUNDING RESULTS -- {model_key}")
    print(f"{'='*60}")
    print(f"  Images processed : {len(pairs)}")
    print(f"  GT boxes total   : {total_gt}")
    print(f"  Predicted boxes  : {total_pred}")
    print(f"  Matched (IoU>{iou_threshold}): {total_matched}")
    print(f"  Precision        : {overall_prec:.1f}%")
    print(f"  Recall           : {overall_rec:.1f}%")
    print(f"  Mean IoU (matched): {overall_iou:.3f}")
    print(f"  Total time       : {int(total_time)}s")
    print(f"  Results          : {out_csv}")
    print(f"{'='*60}\n")


# -- Entry point ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM visual grounding evaluation: locate trash and compare vs YOLO GT"
    )
    parser.add_argument("--model", required=True, choices=["qwen_vl", "qwen_2b"],
                        help="Model key (only Qwen models support grounding)")
    parser.add_argument("--dataset", default=None,
                        help="YOLO dataset root (train/val/valid/test splits)")
    parser.add_argument("--images", default=None,
                        help="Flat images directory (alternative to --dataset)")
    parser.add_argument("--labels", default=None,
                        help="Separate labels directory (with --images)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max images to process (default: 50). Prioritizes labeled images.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, dest="iou_threshold",
                        help="IoU threshold for matching (default: 0.3)")
    parser.add_argument("--out", default="grounding_results",
                        help="Output directory (default: grounding_results)")
    parser.add_argument("--lora-adapter", default=None, dest="lora_adapter",
                        help="Path to saved LoRA adapter directory (e.g. pope_results_v7/qwen_2b_lora)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated images (GT=green, matched=cyan, unmatched=red) to <out>/viz/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_csv = out_dir / f"grounding_{args.model}.csv"

    dataset_path  = Path(args.dataset)      if args.dataset      else None
    images_path   = Path(args.images)       if args.images       else None
    labels_path   = Path(args.labels)       if args.labels       else None
    lora_path     = Path(args.lora_adapter) if args.lora_adapter else None

    if lora_path and not lora_path.exists():
        print(f"ERROR: LoRA adapter not found: {lora_path}")
        raise SystemExit(1)

    pairs = collect_images(dataset_path, images_path, labels_path, args.limit)
    if not pairs:
        print("ERROR: No images found.")
        return

    n_labeled = sum(1 for p in pairs if p["label"].exists() and p["label"].stat().st_size > 0)
    print(f"Selected {len(pairs)} images ({n_labeled} with GT labels)")

    viz_dir = (out_dir / "viz") if args.visualize else None
    run_grounding(args.model, pairs, out_csv, args.iou_threshold,
                  lora_adapter=lora_path, viz_dir=viz_dir)


if __name__ == "__main__":
    main()
