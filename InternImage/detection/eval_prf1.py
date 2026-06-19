"""
Compute per-class and overall P / R / F1 from mmdet test.py --out results.pkl

Usage:
    # Step 1: generate predictions
    torchrun --nproc_per_node=1 test.py <config> <checkpoint> \
        --launcher pytorch --out results.pkl

    # Step 2: compute P/R/F1
    python eval_prf1.py <config> results.pkl --conf 0.3

    # Also accepts --iou (default 0.5) for IoU matching threshold
    # Results saved to eval_results/<pkl_stem>_conf<conf>.csv and .txt
"""

import argparse
import csv
import pickle
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
from collections import defaultdict

import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset
from pycocotools.coco import COCO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="mmdet config file")
    p.add_argument("pkl", help="results.pkl from test.py --out")
    p.add_argument("--conf", type=float, default=0.3,
                   help="confidence threshold (default 0.3)")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for TP matching (default 0.5)")
    p.add_argument("--ann-file", default=None,
                   help="override annotation file (e.g. instances_val.json path)")
    p.add_argument("--img-prefix", default=None,
                   help="override image prefix")
    p.add_argument("--out-dir", default="eval_results",
                   help="directory to save CSV and TXT results (default: eval_results)")
    return p.parse_args()


def iou_single(box_a, box_b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file
    if args.img_prefix:
        cfg.data.test.img_prefix = args.img_prefix
    dataset = build_dataset(cfg.data.test)

    coco = dataset.coco
    cat_ids = dataset.cat_ids
    class_names = dataset.CLASSES
    img_ids = dataset.img_ids
    num_classes = len(class_names)

    with open(args.pkl, "rb") as f:
        results = pickle.load(f)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for idx, img_id in enumerate(img_ids):
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        gt_by_class = defaultdict(list)
        for ann in anns:
            cls_idx = cat_ids.index(ann["category_id"])
            gt_by_class[cls_idx].append(xywh_to_xyxy(ann["bbox"]))

        result = results[idx]
        for cls_idx in range(min(num_classes, len(result))):
            dets = result[cls_idx]
            if len(dets) == 0:
                fn[cls_idx] += len(gt_by_class.get(cls_idx, []))
                continue

            mask = dets[:, 4] >= args.conf
            dets = dets[mask]

            gt_boxes = gt_by_class.get(cls_idx, [])
            matched_gt = set()

            sorted_idx = np.argsort(-dets[:, 4])
            for di in sorted_idx:
                det_box = dets[di, :4]
                best_iou = 0.0
                best_gi = -1
                for gi, gt_box in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    cur_iou = iou_single(det_box, gt_box)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_gi = gi
                if best_iou >= args.iou and best_gi >= 0:
                    tp[cls_idx] += 1
                    matched_gt.add(best_gi)
                else:
                    fp[cls_idx] += 1

            fn[cls_idx] += len(gt_boxes) - len(matched_gt)

    rows = []
    total_tp = total_fp = total_fn = 0
    for c in range(num_classes):
        t, f_p, f_n = tp[c], fp[c], fn[c]
        total_tp += t
        total_fp += f_p
        total_fn += f_n
        p = t / (t + f_p) if (t + f_p) > 0 else 0
        r = t / (t + f_n) if (t + f_n) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        rows.append((class_names[c], t, f_p, f_n, p, r, f1))

    p_total = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r_total = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_total = 2 * p_total * r_total / (p_total + r_total) if (p_total + r_total) > 0 else 0
    rows.append(("TOTAL", total_tp, total_fp, total_fn, p_total, r_total, f1_total))

    header = f"conf={args.conf}  iou={args.iou}  pkl={args.pkl}  time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    col_fmt = f"{'Class':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>8} {'R':>8} {'F1':>8}"
    sep = "-" * 62

    # Print to stdout
    print(f"\n{header}")
    print(col_fmt)
    print(sep)
    for name, t, f_p, f_n, p, r, f1 in rows[:-1]:
        print(f"{name:<20} {t:>6} {f_p:>6} {f_n:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")
    print(sep)
    name, t, f_p, f_n, p, r, f1 = rows[-1]
    print(f"{name:<20} {t:>6} {f_p:>6} {f_n:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")

    # Save to files
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.pkl))[0]
    base = os.path.join(args.out_dir, f"{stem}_conf{args.conf}_iou{args.iou}")

    with open(base + ".txt", "w") as f:
        f.write(header + "\n")
        f.write(col_fmt + "\n")
        f.write(sep + "\n")
        for name, t, f_p, f_n, p, r, f1 in rows[:-1]:
            f.write(f"{name:<20} {t:>6} {f_p:>6} {f_n:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}\n")
        f.write(sep + "\n")
        name, t, f_p, f_n, p, r, f1 = rows[-1]
        f.write(f"{name:<20} {t:>6} {f_p:>6} {f_n:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}\n")

    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "TP", "FP", "FN", "P", "R", "F1"])
        for name, t, f_p, f_n, p, r, f1 in rows:
            w.writerow([name, t, f_p, f_n, round(p, 4), round(r, 4), round(f1, 4)])

    print(f"\nSaved → {base}.txt")
    print(f"Saved → {base}.csv")


if __name__ == "__main__":
    main()
