"""Common detection metrics (mAP@50, mAP@50-95, P, R; global and per class).

Reads YOLO .txt ground truth and predictions and reproduces the Ultralytics
validation computation, so numbers are comparable with the YOLO runs.
See SPEC.md. Only needs numpy (pyyaml for --data).

Example:
    python eval_detections.py --gt val/labels \
        --pred yolo=runs/pred_yolo/labels --pred yolo_vlm=runs/pred_vlm/labels \
        --data data.yaml --out eval_out
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

EPS = 1e-16
IOU_THRS = np.linspace(0.5, 0.95, 10)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_labels(path: Path, with_conf: bool):
    """Return (cls[n], boxes_xyxy[n,4], conf[n]); conf=1 for ground truth."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            need = 6 if with_conf else 5
            if len(parts) < need:
                raise ValueError(f"{path}: expected {need} columns, got '{line}'"
                                 + (" (predictions need a confidence column; "
                                    "save with --save-conf)" if with_conf else ""))
            vals = [float(v) for v in parts[:need]]
            rows.append(vals + ([] if with_conf else [1.0]))
    a = np.array(rows, dtype=float).reshape(-1, 6)
    cls, cx, cy, w, h, conf = a.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return cls.astype(int), xyxy, conf


def box_iou(a, b):
    """IoU matrix between boxes a[n,4] and b[m,4] (xyxy). Scale-invariant, so
    normalised coordinates give the same IoU as pixels."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None] - inter + EPS)


def match_image(gt_cls, gt_box, p_cls, p_box):
    """tp[n_pred, 10]: prediction matches a same-class GT at each IoU threshold
    (one to one, highest IoU first, as in Ultralytics)."""
    tp = np.zeros((len(p_cls), len(IOU_THRS)), dtype=bool)
    if len(p_cls) == 0 or len(gt_cls) == 0:
        return tp
    iou = box_iou(gt_box, p_box) * (gt_cls[:, None] == p_cls[None, :])
    for i, thr in enumerate(IOU_THRS):
        g, p = np.nonzero(iou >= thr)
        if len(g) == 0:
            continue
        m = np.stack([g, p, iou[g, p]], 1)
        if len(m) > 1:
            m = m[np.argsort(-m[:, 2], kind="stable")]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.argsort(-m[:, 2], kind="stable")]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        tp[m[:, 1].astype(int), i] = True
    return tp


def compute_ap(recall, precision):
    """101-point interpolated AP over the precision envelope."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapezoid(np.interp(x, mrec, mpre), x) if hasattr(np, "trapezoid") \
        else np.trapz(np.interp(x, mrec, mpre), x)


def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 * 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def per_class_metrics(tp, conf, pred_cls, target_cls):
    """Same as Ultralytics ap_per_class. Returns dict class -> metrics, for
    every class that has ground truth."""
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    classes = np.unique(target_cls)
    px = np.linspace(0, 1, 1000)
    n_c = len(classes)
    ap = np.zeros((n_c, len(IOU_THRS)))
    p_curve, r_curve = np.zeros((n_c, 1000)), np.zeros((n_c, 1000))
    for ci, c in enumerate(classes):
        sel = pred_cls == c
        n_l, n_p = (target_cls == c).sum(), sel.sum()
        if n_p == 0:
            continue
        fpc = (~tp[sel]).cumsum(0)
        tpc = tp[sel].cumsum(0)
        recall = tpc / (n_l + EPS)
        precision = tpc / (tpc + fpc)
        r_curve[ci] = np.interp(-px, -conf[sel], recall[:, 0], left=0)
        p_curve[ci] = np.interp(-px, -conf[sel], precision[:, 0], left=1)
        for j in range(len(IOU_THRS)):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1 = 2 * p_curve * r_curve / (p_curve + r_curve + EPS)
    i = smooth(f1.mean(0), 0.1).argmax()
    out = {}
    for ci, c in enumerate(classes):
        out[int(c)] = dict(P=float(p_curve[ci, i]), R=float(r_curve[ci, i]),
                           mAP50=float(ap[ci, 0]), mAP50_95=float(ap[ci].mean()),
                           n_gt=int((target_cls == c).sum()))
    return out, float(px[i])


def list_stems(gt_dir: Path, images_dir):
    if images_dir:
        return sorted(p.stem for p in Path(images_dir).rglob("*")
                      if p.suffix.lower() in IMG_EXTS)
    return sorted(p.stem for p in gt_dir.glob("*.txt"))


def evaluate(gt_dir, pred_dir, images_dir=None, names=None):
    """Evaluate one system. Returns {'rows': [...], 'conf_at_max_f1': float}."""
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    stems = list_stems(gt_dir, images_dir)
    if not stems:
        raise SystemExit(f"no images/labels found in {images_dir or gt_dir}")
    tps, confs, pcls, tcls = [], [], [], []
    for s in stems:
        g_cls, g_box, _ = read_labels(gt_dir / f"{s}.txt", with_conf=False)
        p_cls, p_box, p_conf = read_labels(pred_dir / f"{s}.txt", with_conf=True)
        o = np.argsort(-p_conf, kind="stable")  # ties: higher confidence matches first
        p_cls, p_box, p_conf = p_cls[o], p_box[o], p_conf[o]
        tps.append(match_image(g_cls, g_box, p_cls, p_box))
        confs.append(p_conf); pcls.append(p_cls); tcls.append(g_cls)
    tp = np.concatenate(tps); conf = np.concatenate(confs)
    pred_cls = np.concatenate(pcls); target_cls = np.concatenate(tcls)
    if len(target_cls) == 0:
        raise SystemExit("ground truth has no boxes")
    per_cls, conf_thr = per_class_metrics(tp, conf, pred_cls, target_cls)
    names = names or {}
    rows = [dict(class_id=c, name=names.get(c, f"class_{c}"), **m)
            for c, m in per_cls.items()]
    mean = {k: float(np.mean([r[k] for r in rows]))
            for k in ("P", "R", "mAP50", "mAP50_95")}
    rows.append(dict(class_id=-1, name="all", n_gt=int(len(target_cls)), **mean))
    return dict(rows=rows, conf_at_max_f1=conf_thr, n_images=len(stems))


def load_names(data_yaml, names_arg):
    if names_arg:
        return {i: n.strip() for i, n in enumerate(names_arg.split(","))}
    if data_yaml:
        import yaml
        n = yaml.safe_load(Path(data_yaml).read_text()).get("names", {})
        return dict(enumerate(n)) if isinstance(n, list) else {int(k): v for k, v in n.items()}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="Dir with YOLO ground-truth .txt")
    ap.add_argument("--pred", action="append", required=True, metavar="NAME=DIR",
                    help="Predictions dir (YOLO txt with conf). Repeatable.")
    ap.add_argument("--images", help="Images dir (so images without labels count as empty)")
    ap.add_argument("--data", help="data.yaml, for class names")
    ap.add_argument("--names", help="Comma-separated class names")
    ap.add_argument("--out", default="eval_out", help="Output dir")
    args = ap.parse_args()

    names = load_names(args.data, args.names)
    results = {}
    for item in args.pred:
        if "=" not in item:
            ap.error("--pred must be NAME=DIR")
        name, d = item.split("=", 1)
        results[name] = evaluate(args.gt, d, args.images, names)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    fields = ["system", "class_id", "name", "n_gt", "P", "R", "mAP50", "mAP50_95"]
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sysname, res in results.items():
            for r in res["rows"]:
                w.writerow({"system": sysname, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                    for k, v in r.items()}})

    base = next(iter(results))
    print(f"{'system':<14}{'class':<14}{'n_gt':>6}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for sysname, res in results.items():
        for r in res["rows"]:
            print(f"{sysname:<14}{r['name']:<14}{r['n_gt']:>6}{r['P']:>8.3f}{r['R']:>8.3f}"
                  f"{r['mAP50']:>8.3f}{r['mAP50_95']:>10.3f}")
        print(f"  (P y R en confianza {res['conf_at_max_f1']:.3f}, imágenes: {res['n_images']})")
    if len(results) > 1:
        b = {r["name"]: r for r in results[base]["rows"]}
        print(f"\nDiferencia frente a '{base}' (columna all):")
        for sysname, res in list(results.items())[1:]:
            a = next(r for r in res["rows"] if r["name"] == "all")
            print(f"  {sysname}: mAP50 {a['mAP50']-b['all']['mAP50']:+.3f}  "
                  f"mAP50-95 {a['mAP50_95']-b['all']['mAP50_95']:+.3f}  "
                  f"P {a['P']-b['all']['P']:+.3f}  R {a['R']-b['all']['R']:+.3f}")
    print(f"\nGuardado en {out}/metrics.csv y metrics.json")


if __name__ == "__main__":
    main()
