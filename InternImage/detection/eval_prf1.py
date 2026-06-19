"""
Compute per-class and overall P / R / F1 from mmdet test.py --out results.pkl

Usage:
    # Step 1: generate predictions
    torchrun --nproc_per_node=1 test.py <config> <checkpoint> \
        --launcher pytorch --out results.pkl

    # Step 2: compute P/R/F1
    python eval_prf1.py <config> results.pkl --conf 0.3

    # Also accepts --iou (default 0.5) for IoU matching threshold
"""

import argparse
import pickle
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

    print(f"\nconf={args.conf}  iou={args.iou}")
    print(f"{'Class':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 62)

    total_tp = total_fp = total_fn = 0
    for c in range(num_classes):
        t, f_p, f_n = tp[c], fp[c], fn[c]
        total_tp += t
        total_fp += f_p
        total_fn += f_n
        p = t / (t + f_p) if (t + f_p) > 0 else 0
        r = t / (t + f_n) if (t + f_n) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{class_names[c]:<20} {t:>6} {f_p:>6} {f_n:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print("-" * 62)
    print(f"{'TOTAL':<20} {total_tp:>6} {total_fp:>6} {total_fn:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")


if __name__ == "__main__":
    main()
