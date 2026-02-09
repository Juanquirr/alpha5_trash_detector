import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

PURPLE = "#8000ff"
BAR_FORMAT = (
    "{desc:<28} |{bar}| {percentage:6.2f}% "
    "({n_fmt}/{total_fmt}) [{elapsed}<{remaining}, {rate_fmt}]"
)


def apply_tta_augmentations(image):
    """
    Generate augmented versions of image for TTA
    
    Returns:
        List of (augmented_image, transform_info) tuples
    """
    augmentations = []
    h, w = image.shape[:2]
    
    # Original
    augmentations.append((image.copy(), {"type": "original", "params": None}))
    
    # Horizontal flip
    aug_h = cv2.flip(image, 1)
    augmentations.append((aug_h, {"type": "flip_h", "params": {"width": w}}))
    
    # Vertical flip
    aug_v = cv2.flip(image, 0)
    augmentations.append((aug_v, {"type": "flip_v", "params": {"height": h}}))
    
    # Both flips
    aug_hv = cv2.flip(image, -1)
    augmentations.append((aug_hv, {"type": "flip_hv", "params": {"width": w, "height": h}}))
    
    # Scale up 1.1x
    aug_scale = cv2.resize(image, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_LINEAR)
    augmentations.append((aug_scale, {"type": "scale", "params": {"scale": 1.1, "orig_w": w, "orig_h": h}}))
    
    return augmentations


def reverse_transform(boxes, transform_info):
    """
    Reverse augmentation transformation on bounding boxes
    
    Args:
        boxes: Array of [x1, y1, x2, y2] bounding boxes
        transform_info: Dict with transformation type and parameters
    
    Returns:
        Transformed boxes in original image coordinates
    """
    if len(boxes) == 0:
        return boxes
    
    boxes = np.array(boxes)
    transform_type = transform_info["type"]
    params = transform_info["params"]
    
    if transform_type == "original":
        return boxes
    
    elif transform_type == "flip_h":
        # Horizontal flip: x' = width - x
        w = params["width"]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    
    elif transform_type == "flip_v":
        # Vertical flip: y' = height - y
        h = params["height"]
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    
    elif transform_type == "flip_hv":
        # Both flips
        w = params["width"]
        h = params["height"]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    
    elif transform_type == "scale":
        # Scale back
        scale = params["scale"]
        boxes = boxes / scale
    
    return boxes


def weighted_boxes_fusion_simple(all_boxes, all_scores, all_classes, iou_thresh=0.5):
    """
    Simple WBF implementation for TTA
    
    Args:
        all_boxes: List of box arrays from different augmentations
        all_scores: List of confidence arrays
        all_classes: List of class arrays
        iou_thresh: IoU threshold for fusion
    
    Returns:
        Fused boxes, scores, classes
    """
    if not all_boxes:
        return np.array([]), np.array([]), np.array([])
    
    # Concatenate all detections
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    
    if len(boxes) == 0:
        return boxes, scores, classes
    
    # Simple NMS-based fusion (you can replace with proper WBF)
    keep = []
    order = scores.argsort()[::-1]
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Compute IoU
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
        
        # Keep boxes with IoU < threshold
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    
    return boxes[keep], scores[keep], classes[keep]


def process_image_tta(
    model: YOLO,
    img_path: Path,
    out_dir: Path,
    conf: float,
    iou: float,
    device: str,
    imgsz: int,
    tta_iou: float,
    pbar: tqdm = None
):
    """
    Process image with Test-Time Augmentation
    
    Args:
        model: YOLO model
        img_path: Path to image
        out_dir: Output directory
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: Device
        imgsz: Input size
        tta_iou: IoU threshold for TTA fusion
        pbar: Progress bar
    """
    img = cv2.imread(str(img_path))
    if img is None:
        tqdm.write(f"⚠️  Skipping: {img_path}")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if pbar:
        pbar.set_description_str(f"TTA {img_path.name}")
    
    # Generate augmentations
    augmentations = apply_tta_augmentations(img)
    
    all_boxes, all_scores, all_classes = [], [], []
    
    # Detect on each augmentation
    for aug_img, transform_info in augmentations:
        results = model.predict(
            aug_img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            device=device
        )
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            
            # Reverse transformation
            boxes_orig = reverse_transform(boxes, transform_info)
            
            all_boxes.append(boxes_orig)
            all_scores.append(scores)
            all_classes.append(clss)
    
    # Fusion
    if all_boxes:
        final_boxes, final_scores, final_classes = weighted_boxes_fusion_simple(
            all_boxes, all_scores, all_classes, iou_thresh=tta_iou
        )
    else:
        final_boxes = np.array([])
        final_scores = np.array([])
        final_classes = np.array([])
    
    tqdm.write(f"  ✓ {img_path.name}: {len(final_boxes)} detections (TTA from {len(augmentations)} augs)")
    
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
    output_path = out_dir / f"tta_{img_path.stem}.jpg"
    cv2.imwrite(str(output_path), img_out)


def iter_images(source: Path, recursive: bool = False):
    """Iterate over images"""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", "."}
    
    if source.is_file():
        return [source] if source.suffix.lower() in supported else []
    
    if recursive:
        return sorted([p for ext in supported for p in source.rglob(f"*{ext}")])
    else:
        return sorted([p for ext in supported for p in source.glob(f"*{ext}")])


def build_args():
    """Build argument parser"""
    p = argparse.ArgumentParser(
        description="Test-Time Augmentation for YOLO object detection"
    )
    
    p.add_argument("source", type=str, help="Input image or directory")
    p.add_argument("model", type=str, help="Path to YOLO model (.pt)")
    p.add_argument("--out_dir", type=str, default="tta_results", help="Output directory")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU for NMS")
    p.add_argument("--tta_iou", type=float, default=0.5, help="IoU for TTA fusion")
    p.add_argument("--imgsz", type=int, default=640, help="Input size")
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
    print(f"✓ TTA with 5 augmentations (original + 4 transforms)")
    print(f"✓ Confidence: {args.conf}")
    print(f"✓ TTA IoU fusion: {args.tta_iou}")
    
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
            process_image_tta(
                model=model,
                img_path=img_path,
                out_dir=out_dir,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                imgsz=args.imgsz,
                tta_iou=args.tta_iou,
                pbar=pbar
            )
            pbar.update(1)
    finally:
        pbar.close()
    
    print(f"\n✓ TTA complete! Results in {out_dir}")


if __name__ == '__main__':
    main()
