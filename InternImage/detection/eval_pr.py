"""
Precision / Recall for MMDetection (InternImage) models, comparable to YOLO.

COCO eval (the default MMDet metric) reports mAP variants but not the
single-threshold precision/recall that YOLO prints. This script runs the
trained detector over the val split, greedily matches predictions to GT at a
fixed IoU, sweeps the confidence threshold to find the best-F1 operating point
(same idea YOLO uses) and reports global + per-class P/R/F1 there.

Run from InternImage/detection/ (so mmdet_custom datasets register and ./data
paths resolve):

    python eval_pr.py \
        --config work_dirs/cascade_internimage_l_alpha5_3x/cascade_internimage_l_alpha5_3x.py \
        --checkpoint work_dirs/cascade_internimage_l_alpha5_3x/best_bbox_mAP_epoch_33.pth

    python eval_pr.py \
        --config work_dirs/dino_4scale_internimage_l_alpha5_3x/dino_4scale_internimage_l_alpha5_3x.py \
        --checkpoint work_dirs/dino_4scale_internimage_l_alpha5_3x/best_bbox_mAP_epoch_34.pth

Defaults to the val split declared in the config. mmdet 2.x API.
"""
import argparse
import os.path as osp
from collections import defaultdict

import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO

import mmdet_custom  # noqa: F401  registers InternImage backbone, DINO head, Alpha5Dataset


def iou_xyxy(a, boxes):
    """IoU of box a [x1,y1,x2,y2] against array boxes (N,4)."""
    if len(boxes) == 0:
        return np.zeros((0,))
    x1 = np.maximum(a[0], boxes[:, 0])
    y1 = np.maximum(a[1], boxes[:, 1])
    x2 = np.minimum(a[2], boxes[:, 2])
    y2 = np.minimum(a[3], boxes[:, 3])
    w = np.clip(x2 - x1, 0, None)
    h = np.clip(y2 - y1, 0, None)
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def get_bbox_result(result):
    """mmdet 2.x: bbox-only -> list; mask models -> (bbox, segm) tuple."""
    if isinstance(result, tuple):
        return result[0]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU match threshold")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    args = ap.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    dcfg = getattr(cfg.data, args.split)
    ann_file = dcfg.ann_file
    img_prefix = dcfg.img_prefix

    model = init_detector(args.config, args.checkpoint, device=args.device)
    classes = getattr(model, "CLASSES", None)
    if not classes:
        classes = list(dcfg.get("classes", None) or cfg.get("classes", None) or [])
    classes = list(classes)
    if not classes:
        raise SystemExit("Could not resolve class names from model or config.")
    n_cls = len(classes)
    print(f"Classes ({n_cls}): {classes}")

    coco = COCO(ann_file)
    cat_id_to_idx = {}
    for c in coco.loadCats(coco.getCatIds()):
        if c["name"] in classes:
            cat_id_to_idx[c["id"]] = classes.index(c["name"])
    if not cat_id_to_idx:
        raise SystemExit(
            "No COCO category name matches model classes. "
            f"COCO cats={[c['name'] for c in coco.loadCats(coco.getCatIds())]} "
            f"model classes={classes}")

    # collect predictions and GT per image
    # preds[img]: list of (cls_idx, score, box)
    # gts[img]:   list of (cls_idx, box)
    all_preds, all_gts = {}, {}
    n_gt_per_cls = np.zeros(n_cls, dtype=int)

    img_ids = coco.getImgIds()
    for i, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        img_path = osp.join(img_prefix, info["file_name"])

        gts = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            if ann["category_id"] not in cat_id_to_idx:
                continue
            x, y, w, h = ann["bbox"]
            ci = cat_id_to_idx[ann["category_id"]]
            gts.append((ci, np.array([x, y, x + w, y + h], dtype=float)))
            n_gt_per_cls[ci] += 1
        all_gts[img_id] = gts

        bbox_result = get_bbox_result(inference_detector(model, img_path))
        if len(bbox_result) != n_cls:
            raise SystemExit(
                f"Model returned {len(bbox_result)} class arrays but {n_cls} classes "
                "expected. Check the config head num_classes matches CLASSES.")
        preds = []
        for ci in range(n_cls):
            arr = bbox_result[ci]
            for det in arr:
                preds.append((ci, float(det[4]), det[:4].astype(float)))
        all_preds[img_id] = preds

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(img_ids)} images")

    # sweep confidence threshold, greedy match per class per image @ IoU
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best = {"f1": -1}
    for thr in thresholds:
        tp = np.zeros(n_cls, dtype=int)
        fp = np.zeros(n_cls, dtype=int)
        for img_id in img_ids:
            gts = all_gts[img_id]
            gt_boxes_by_cls = defaultdict(list)
            for ci, box in gts:
                gt_boxes_by_cls[ci].append(box)
            used = {ci: np.zeros(len(v), dtype=bool) for ci, v in gt_boxes_by_cls.items()}

            dets = [p for p in all_preds[img_id] if p[1] >= thr]
            dets.sort(key=lambda p: -p[1])
            for ci, score, box in dets:
                gboxes = gt_boxes_by_cls.get(ci, [])
                if len(gboxes) == 0:
                    fp[ci] += 1
                    continue
                ious = iou_xyxy(box, np.array(gboxes))
                order = np.argsort(-ious)
                hit = False
                for j in order:
                    if ious[j] < args.iou:
                        break
                    if not used[ci][j]:
                        used[ci][j] = True
                        tp[ci] += 1
                        hit = True
                        break
                if not hit:
                    fp[ci] += 1

        fn = n_gt_per_cls - tp
        P = tp.sum() / max(tp.sum() + fp.sum(), 1)
        R = tp.sum() / max(n_gt_per_cls.sum(), 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        if F1 > best["f1"]:
            best = {"f1": F1, "thr": float(thr), "P": P, "R": R,
                    "tp": tp.copy(), "fp": fp.copy(), "fn": fn.copy()}

    thr = best["thr"]
    print("\n================ BEST-F1 OPERATING POINT ================")
    print(f"conf threshold = {thr:.2f}   (IoU match = {args.iou})")
    print(f"GLOBAL   P={best['P']:.3f}  R={best['R']:.3f}  F1={best['f1']:.3f}")
    print(f"\n{'class':<18}{'P':>8}{'R':>8}{'F1':>8}{'GT':>7}")
    tp, fp, fn = best["tp"], best["fp"], best["fn"]
    for ci, name in enumerate(classes):
        p = tp[ci] / max(tp[ci] + fp[ci], 1)
        r = tp[ci] / max(tp[ci] + fn[ci], 1)
        f = 2 * p * r / max(p + r, 1e-9)
        print(f"{name:<18}{p:>8.3f}{r:>8.3f}{f:>8.3f}{n_gt_per_cls[ci]:>7}")
    print("========================================================")


if __name__ == "__main__":
    main()
