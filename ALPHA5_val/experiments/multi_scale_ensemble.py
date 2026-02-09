import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

PURPLE = "#8000ff"
BAR_FORMAT = "{desc:<28} |{bar}| {percentage:6.2f}% ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}, {rate_fmt}]"


def nms_boxes(boxes, scores, classes, iou_thresh=0.5):
    """Apply NMS"""
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    keep = []
    order = scores.argsort()[::-1]
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_others - inter
        
        iou = inter / (union + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    
    return boxes[keep], scores[keep], classes[keep]


def process_image_multiscale(
    model: YOLO,
    img_path: Path,
    out_dir: Path,
    conf: float,
    iou: float,
    device: str,
    scales: list,
    nms_thresh: float,
    pbar: tqdm = None
):
    """Process image at multiple scales"""
    img = cv2.imread(str(img_path))
    if img is None:
        tqdm.write(f"⚠️  Skipping: {img_path}")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if pbar:
        pbar.set_description_str(f"MultiScale {img_path.name}")
    
    all_boxes, all_scores, all_classes = [], [], []
    
    # Detect at each scale
    for scale in scales:
        results = model.predict(
            img,
            conf=conf,
            iou=iou,
            imgsz=scale,
            verbose=False,
            device=device
        )
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(clss)
    
    # Fuse results
    if all_boxes:
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)
        
        final_boxes, final_scores, final_classes = nms_boxes(
            boxes, scores, classes, iou_thresh=nms_thresh
        )
    else:
        final_boxes = np.array([])
        final_scores = np.array([])
        final_classes = np.array([])
    
    tqdm.write(f"  ✓ {img_path.name}: {len(final_boxes)} detections (scales: {scales})")
    
    # Annotate
    annotator = Annotator(img, line_width=2, example=model.names)
    
    for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
        name = model.names.get(int(cls_id), str(int(cls_id)))
        annotator.box_label(
            [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            f"{name} {score:.2f}",
            color=colors(int(cls_id), bgr=True)
        )
    
    img_out = annotator.result()
    output_path = out_dir / f"multiscale_{img_path.stem}.jpg"
    cv2.imwrite(str(output_path), img_out)


def iter_images(source: Path, recursive: bool = False):
    """Iterate over images"""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    
    if source.is_file():
        return [source] if source.suffix.lower() in supported else []
    
    if recursive:
        return sorted([p for ext in supported for p in source.rglob(f"*{ext}")])
    else:
        return sorted([p for ext in supported for p in source.glob(f"*{ext}")])


def build_args():
    """Build argument parser"""
    p = argparse.ArgumentParser(
        description="Multi-scale inference for YOLO detection"
    )
    
    p.add_argument("source", type=str, help="Input image or directory")
    p.add_argument("model", type=str, help="Path to YOLO model (.pt)")
    p.add_argument("--out_dir", type=str, default="multiscale_results", help="Output directory")
    p.add_argument("--scales", type=int, nargs='+', default=[640, 960, 1280],
                   help="Scales for multi-scale inference (default: 640 960 1280)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    p.add_argument("--nms_thresh", type=float, default=0.5, help="NMS threshold for fusion")
    p.add_argument("--device", type=str, default="cuda:0", help="Device")
    p.add_argument("--recursive", action="store_true", help="Recursive search")
    
    return p.parse_args()


def main():
    """Main entry point"""
    args = build_args()
    
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    
    model = YOLO(args.model)
    images = iter_images(source, args.recursive)
    
    if not images:
        raise SystemExit(f"❌ No images found in: {source}")
    
    print(f"✓ Found {len(images)} image(s)")
    print(f"✓ Scales: {args.scales}")
    print(f"✓ Confidence: {args.conf}")
    print(f"✓ NMS threshold: {args.nms_thresh}")
    
    pbar = tqdm(
        total=len(images),
        desc="Images processed",
        unit="img",
        bar_format=BAR_FORMAT,
        colour=PURPLE,
        dynamic_ncols=True
    )
    
    try:
        for img_path in images:
            process_image_multiscale(
                model=model,
                img_path=img_path,
                out_dir=out_dir,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                scales=args.scales,
                nms_thresh=args.nms_thresh,
                pbar=pbar
            )
            pbar.update(1)
    finally:
        pbar.close()
    
    print(f"\n✓ Multi-scale inference complete! Results in {out_dir}")


if __name__ == '__main__':
    main()
