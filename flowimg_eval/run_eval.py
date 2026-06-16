"""
FlowIMG detection-only evaluation (class-agnostic).

All model predictions (any class) → treated as "trash".
All FlowIMG GT annotations (bottle) → treated as "trash".

Metrics computed:
  mAP@50       — COCO-style 101-point interpolation, IoU=0.50
  mAP@50:95    — mean AP across IoU thresholds 0.50→0.95 (step 0.05)
  Precision    — at conf=CONF_THRESHOLD operating point
  Recall       — at conf=CONF_THRESHOLD operating point
  F1           — harmonic mean of P and R

Usage:
  python run_eval.py

Edit config.py to add/remove models or change settings.
"""

import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config import (
    BATCH_SIZE,
    CONF_THRESHOLD,
    DEVICE,
    FLOWIMG_TEST_ANNOTS,
    FLOWIMG_TEST_IMAGES,
    IMGSZ,
    MODELS,
    N_SAMPLE_IMAGES,
    RESULTS_DIR,
    SAVE_PREDICTIONS,
)


# ─── GT parsing ───────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path: Path) -> np.ndarray:
    """Return [N, 4] xyxy boxes from a Pascal VOC XML file."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append([
            float(b.find("xmin").text),
            float(b.find("ymin").text),
            float(b.find("xmax").text),
            float(b.find("ymax").text),
        ])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def load_all_gt() -> dict:
    """Load all test GT annotations. Returns dict: stem -> np.ndarray [N,4]."""
    xml_paths = sorted(FLOWIMG_TEST_ANNOTS.glob("*.xml"))
    gts = {p.stem: parse_voc_xml(p) for p in xml_paths}
    total_instances = sum(len(v) for v in gts.values())
    print(f"GT loaded: {len(gts)} images | {total_instances} instances")
    return gts


# ─── IoU & metric computation ─────────────────────────────────────────────────

def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix [N, M] for xyxy boxes."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    ix1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-7)


def compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point interpolated AP (COCO style)."""
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recalls >= t
        ap += float(np.max(precisions[mask])) if mask.any() else 0.0
    return ap / 101.0


def match_and_compute_ap(image_preds: dict, image_gts: dict, iou_thr: float):
    """
    Greedily match predictions to GT across all images at given IoU threshold.
    Each GT box can only be matched once (highest-confidence pred wins).

    Returns: (ap, precision, recall) all as float.
    """
    all_dets = []  # list of (conf, is_tp: bool)
    total_gt = sum(len(v) for v in image_gts.values())

    for img_id, gt_boxes in image_gts.items():
        preds = image_preds.get(img_id)
        n_pred = len(preds["boxes"]) if preds else 0

        if n_pred == 0:
            continue

        pred_boxes = preds["boxes"]
        pred_confs = preds["confs"]

        # Sort predictions by confidence (descending) for greedy matching
        order = np.argsort(-pred_confs)
        pred_boxes = pred_boxes[order]
        pred_confs = pred_confs[order]

        matched_gt = set()

        if len(gt_boxes) > 0:
            iou_mat = box_iou_matrix(pred_boxes, gt_boxes)  # [N_pred, N_gt]
        else:
            iou_mat = np.zeros((n_pred, 0))

        for p in range(n_pred):
            conf = float(pred_confs[p])

            if len(gt_boxes) == 0:
                all_dets.append((conf, False))
                continue

            ious = iou_mat[p].copy()
            for m in matched_gt:
                ious[m] = -1.0  # mask already-matched GT

            best_gt = int(np.argmax(ious))
            is_tp = bool(ious[best_gt] >= iou_thr)
            if is_tp:
                matched_gt.add(best_gt)

            all_dets.append((conf, is_tp))

    if not all_dets or total_gt == 0:
        return 0.0, 0.0, 0.0

    # Sort all detections globally by confidence (descending)
    all_dets.sort(key=lambda x: -x[0])

    tp = np.cumsum([d[1] for d in all_dets], dtype=float)
    fp = np.cumsum([not d[1] for d in all_dets], dtype=float)
    recalls    = tp / total_gt
    precisions = tp / (tp + fp)

    ap = compute_ap_101(recalls, precisions)
    return ap, float(precisions[-1]), float(recalls[-1])


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(model: YOLO, valid_stems: list, img_dict: dict) -> dict:
    """Run model.predict() in batches. Returns dict: stem -> {boxes, confs}."""
    image_preds = {}

    for i in range(0, len(valid_stems), BATCH_SIZE):
        batch_stems = valid_stems[i: i + BATCH_SIZE]
        batch_paths = [str(img_dict[s]) for s in batch_stems]

        results = model.predict(
            batch_paths,
            conf=CONF_THRESHOLD,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False,
        )

        for stem, res in zip(batch_stems, results):
            if res.boxes is not None and len(res.boxes):
                image_preds[stem] = {
                    "boxes": res.boxes.xyxy.cpu().numpy(),
                    "confs": res.boxes.conf.cpu().numpy(),
                }
            else:
                image_preds[stem] = {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "confs": np.zeros(0, dtype=np.float32),
                }

        done = min(i + BATCH_SIZE, len(valid_stems))
        print(f"  Inference [{done}/{len(valid_stems)}]", end="\r")

    print()
    return image_preds


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate_model(model_name: str, model_path: Path, image_gts: dict) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Model  : {model_name}")
    print(f"  Weights: {model_path}")

    model = YOLO(str(model_path))

    img_paths  = sorted(FLOWIMG_TEST_IMAGES.glob("*.jpg"))
    img_dict   = {p.stem: p for p in img_paths}
    valid_stems = sorted(set(img_dict) & set(image_gts))

    # Inference
    image_preds = run_inference(model, valid_stems, img_dict)

    # Metrics at IoU 0.50 → 0.95
    iou_thrs = np.arange(0.50, 1.00, 0.05)
    aps = []
    map50_p, map50_r = 0.0, 0.0

    for idx, thr in enumerate(iou_thrs):
        ap, p, r = match_and_compute_ap(image_preds, image_gts, float(thr))
        aps.append(ap)
        if idx == 0:
            map50_p, map50_r = p, r  # P/R at IoU=0.50

    map50    = aps[0]
    map50_95 = float(np.mean(aps))
    f1       = 2 * map50_p * map50_r / (map50_p + map50_r + 1e-7)
    n_preds  = sum(len(v["confs"]) for v in image_preds.values())
    total_gt = sum(len(v) for v in image_gts.values())

    # Save sample prediction images for presentation
    if SAVE_PREDICTIONS and N_SAMPLE_IMAGES > 0:
        save_dir = RESULTS_DIR / "predictions" / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        samples = valid_stems[:N_SAMPLE_IMAGES]
        model.predict(
            [str(img_dict[s]) for s in samples],
            conf=CONF_THRESHOLD,
            imgsz=IMGSZ,
            device=DEVICE,
            save=True,
            project=str(RESULTS_DIR / "predictions"),
            name=model_name,
            exist_ok=True,
            verbose=False,
        )
        print(f"  Sample images saved → {save_dir}")

    return {
        "model":        model_name,
        "images":       len(valid_stems),
        "gt_instances": total_gt,
        "predictions":  n_preds,
        "mAP50":        round(map50,    4),
        "mAP50_95":     round(map50_95, 4),
        "precision":    round(map50_p,  4),
        "recall":       round(map50_r,  4),
        "f1":           round(f1,       4),
    }


# ─── Output ───────────────────────────────────────────────────────────────────

def print_table(all_results: list):
    cols   = ["model", "images", "gt_instances", "predictions", "mAP50", "mAP50_95", "precision", "recall", "f1"]
    widths = [28,       8,        14,              13,            8,       10,          11,          8,        8]
    header = "".join(str(c).ljust(w) for c, w in zip(cols, widths))
    sep    = "─" * sum(widths)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in all_results:
        row = "".join(str(r[c]).ljust(w) for c, w in zip(cols, widths))
        print(row)
    print(sep)


def save_csv(all_results: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"flowimg_eval_{ts}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV saved → {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FlowIMG Detection-Only Evaluation (class-agnostic)")
    print("=" * 60)
    print(f"Dataset : {FLOWIMG_TEST_IMAGES}")
    print(f"Models  : {list(MODELS.keys())}")
    print(f"Conf    : {CONF_THRESHOLD}  |  IoU: 0.50–0.95  |  ImgSz: {IMGSZ}")

    image_gts  = load_all_gt()
    all_results = []

    for model_name, model_path in MODELS.items():
        if not Path(model_path).exists():
            print(f"\n[SKIP] {model_name}: weights not found → {model_path}")
            continue
        result = evaluate_model(model_name, model_path, image_gts)
        all_results.append(result)
        print(
            f"  → mAP50={result['mAP50']}  "
            f"mAP50:95={result['mAP50_95']}  "
            f"P={result['precision']}  "
            f"R={result['recall']}  "
            f"F1={result['f1']}"
        )

    if all_results:
        print_table(all_results)
        save_csv(all_results)
    else:
        print("\nNo models evaluated (check weights paths in config.py).")


if __name__ == "__main__":
    main()
