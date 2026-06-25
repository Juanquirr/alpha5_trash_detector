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
import json
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
    ap.add_argument("--out", default=None, help="Save results to JSON file (e.g. cascade_pr.json)")
    args = ap.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    dcfg = getattr(cfg.data, args.split)
    ann_file = dcfg.ann_file
    img_prefix = dcfg.img_prefix

    model = init_detector(args.config, args.checkpoint, device=args.device)
    model_classes = list(getattr(model, "CLASSES", None) or [])

    coco = COCO(ann_file)
    coco_cats = coco.loadCats(coco.getCatIds())
    # Eval classes come from the GT (the categories actually present in the
    # dataset), NOT from the model head. The head may declare extra classes
    # that were never trained (e.g. plastic_fragment with 0 instances).
    classes = [c["name"] for c in sorted(coco_cats, key=lambda c: c["id"])]
    n_cls = len(classes)
    name_to_idx = {n: i for i, n in enumerate(classes)}
    catid_to_idx = {c["id"]: name_to_idx[c["name"]] for c in coco_cats}
    print(f"Eval classes ({n_cls}, from GT): {classes}")
    if model_classes:
        extra = [n for n in model_classes if n not in name_to_idx]
        if extra:
            print(f"Model declares extra classes absent from GT (ignored): {extra}")

    # collect predictions and GT per image
    all_preds, all_gts = {}, {}
    n_gt_per_cls = np.zeros(n_cls, dtype=int)

    img_ids = coco.getImgIds()
    for i, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        img_path = osp.join(img_prefix, info["file_name"])

        gts = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            if ann["category_id"] not in catid_to_idx:
                continue
            x, y, w, h = ann["bbox"]
            ci = catid_to_idx[ann["category_id"]]
            gts.append((ci, np.array([x, y, x + w, y + h], dtype=float)))
            n_gt_per_cls[ci] += 1
        all_gts[img_id] = gts

        bbox_result = get_bbox_result(inference_detector(model, img_path))
        if model_classes and len(bbox_result) != len(model_classes):
            raise SystemExit(
                f"Model returned {len(bbox_result)} class arrays but CLASSES has "
                f"{len(model_classes)}. Config head mismatch.")
        preds = []
        for mi, arr in enumerate(bbox_result):
            # map model output index -> eval class by name; skip untrained extras
            if model_classes:
                name = model_classes[mi]
                if name not in name_to_idx:
                    continue
                ci = name_to_idx[name]
            else:
                if mi >= n_cls:
                    continue
                ci = mi
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
    per_class = []
    for ci, name in enumerate(classes):
        p = tp[ci] / max(tp[ci] + fp[ci], 1)
        r = tp[ci] / max(tp[ci] + fn[ci], 1)
        f = 2 * p * r / max(p + r, 1e-9)
        print(f"{name:<18}{p:>8.3f}{r:>8.3f}{f:>8.3f}{n_gt_per_cls[ci]:>7}")
        per_class.append({"class": name, "P": round(p, 4), "R": round(r, 4),
                          "F1": round(f, 4), "GT": int(n_gt_per_cls[ci])})
    print("========================================================")

    if args.out:
        result = {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "iou_threshold": args.iou,
            "best_conf_threshold": thr,
            "global": {"P": round(best["P"], 4), "R": round(best["R"], 4),
                       "F1": round(best["f1"], 4)},
            "per_class": per_class,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
