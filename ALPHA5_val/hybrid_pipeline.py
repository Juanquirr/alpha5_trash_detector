import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

from wbf_utils import compute_iou_xyxy, weighted_boxes_fusion
from crop_utils import UniformCrops, draw_crop_grid


IMG_EXTS = {".jpg", ".jpeg", ".png"}
PURPLE = "#8000ff"
GREEN = "#00ff00"
BAR_FORMAT = (
    "{desc:<28} "
    "|{bar}| "
    "{percentage:6.2f}% "
    "({n_fmt}/{total_fmt}) "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)


def iter_images(source: Path, recursive: bool):
    """List all image files from source path."""
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXTS else []
    if source.is_dir():
        it = source.rglob("*") if recursive else source.iterdir()
        return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS])
    return []


def full_image_inference(model: YOLO, img: np.ndarray, conf: float, device: str):
    """Run inference on full image without crops."""
    results = model.predict(img, device=device, conf=conf, verbose=False)
    r = results[0]
    
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])
    
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    
    return boxes, scores, classes


def crops_inference(model: YOLO, img: np.ndarray, crops_number: int, overlap: float,
                   conf: float, iou: float, device: str, pbar_crops: tqdm = None):
    """Run inference using crops with WBF fusion."""
    cropper = UniformCrops(overlap_ratio=overlap)
    crops, coords = cropper.crop(img, crops_number=crops_number)
    
    if pbar_crops is not None:
        pbar_crops.reset(total=len(crops))
    
    all_boxes, all_scores, all_classes = [], [], []
    
    for crop, (x_min, y_min, _, _) in zip(crops, coords):
        results = model.predict(crop, device=device, conf=conf, verbose=False)
        r = results[0]
        
        if pbar_crops is not None:
            pbar_crops.update(1)
        
        if r.boxes is None or len(r.boxes) == 0:
            continue
        
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        
        # Transform to global coordinates
        for b, s, c in zip(boxes, scores, clss):
            if float(s) < conf:
                continue
            gx1 = float(b[0] + x_min)
            gy1 = float(b[1] + y_min)
            gx2 = float(b[2] + x_min)
            gy2 = float(b[3] + y_min)
            all_boxes.append([gx1, gy1, gx2, gy2])
            all_scores.append(float(s))
            all_classes.append(int(c))
    
    if not all_boxes:
        return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])
    
    boxes = np.array(all_boxes, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    classes = np.array(all_classes, dtype=np.int32)
    
    # Apply WBF
    boxes, scores, classes = weighted_boxes_fusion(
        boxes, scores, classes,
        iou_thres=iou,
        skip_box_thr=conf
    )
    
    return boxes, scores, classes, coords


def smart_filter_crops(crops_boxes, crops_scores, crops_classes,
                      full_boxes, full_scores, full_classes,
                      high_iou_threshold: float = 0.85,
                      suspect_iou_threshold: float = 0.3):
    """
    Smart filtering: keep crops detections that are either:
    1. Validated by full (high IoU) - same object seen by both
    2. New small objects (no overlap) - detected only by crops
    
    Discard only suspicious medium-overlap detections (likely fragments).
    
    Args:
        crops_boxes, crops_scores, crops_classes: Crops detections
        full_boxes, full_scores, full_classes: Full-image detections
        high_iou_threshold: IoU >= this → validated detection (keep)
        suspect_iou_threshold: IoU in [this, high] → suspicious fragment (discard)
        
    Returns:
        Filtered crops detections
    """
    if len(crops_boxes) == 0:
        return crops_boxes, crops_scores, crops_classes
    
    if len(full_boxes) == 0:
        # No full detections, keep all crops (they found small objects)
        return crops_boxes, crops_scores, crops_classes
    
    valid_indices = []
    
    for i, (crop_box, crop_cls) in enumerate(zip(crops_boxes, crops_classes)):
        max_iou = 0.0
        
        # Find max IoU with same-class full detections
        for full_box, full_cls in zip(full_boxes, full_classes):
            if crop_cls != full_cls:
                continue
            iou = compute_iou_xyxy(crop_box, full_box)
            max_iou = max(max_iou, iou)
        
        # Decision logic
        if max_iou >= high_iou_threshold:
            # HIGH overlap: validated by full → KEEP
            valid_indices.append(i)
        elif max_iou < suspect_iou_threshold:
            # NO overlap: new small object → KEEP
            valid_indices.append(i)
        # else: medium overlap (suspect fragment) → DISCARD
    
    if not valid_indices:
        return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])
    
    return (
        crops_boxes[valid_indices],
        crops_scores[valid_indices],
        crops_classes[valid_indices]
    )


def merge_detections(full_boxes, full_scores, full_classes,
                    crops_boxes, crops_scores, crops_classes,
                    merge_iou: float = 0.5):
    """Merge full and filtered crops detections using WBF."""
    # Handle empty arrays properly
    if len(full_boxes) == 0 and len(crops_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])
    elif len(full_boxes) == 0:
        all_boxes, all_scores, all_classes = crops_boxes, crops_scores, crops_classes
    elif len(crops_boxes) == 0:
        all_boxes, all_scores, all_classes = full_boxes, full_scores, full_classes
    else:
        all_boxes = np.concatenate([full_boxes, crops_boxes], axis=0)
        all_scores = np.concatenate([full_scores, crops_scores], axis=0)
        all_classes = np.concatenate([full_classes, crops_classes], axis=0)
    
    # Final WBF to merge overlapping detections
    return weighted_boxes_fusion(
        all_boxes, all_scores, all_classes,
        iou_thres=merge_iou,
        skip_box_thr=0.0
    )


def save_annotated_image(img, boxes, scores, classes, model, save_path):
    """Save image with bounding boxes annotations."""
    img_out = img.copy()
    if len(boxes) > 0:
        annotator = Annotator(img_out, line_width=2, example=model.names)
        for box, score, cls_id in zip(boxes, scores, classes):
            name = model.names.get(int(cls_id), str(int(cls_id)))
            annotator.box_label(box, f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
        img_out = annotator.result()
    cv2.imwrite(str(save_path), img_out)


def process_image_hybrid(model: YOLO, img_path: Path, out_dir: Path,
                        conf: float, device: str,
                        crops_number: int, overlap: float, crops_iou: float,
                        high_iou: float, suspect_iou: float, merge_iou: float,
                        save_intermediate: bool, draw_grid: bool, pbar_crops: tqdm = None):
    """Process image using hybrid two-stage pipeline."""
    img = cv2.imread(str(img_path))
    if img is None:
        tqdm.write(f"⚠️  Skip: {img_path}")
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Full-image inference
    full_boxes, full_scores, full_classes = full_image_inference(model, img, conf, device)
    
    # Stage 2: Crops inference
    crops_boxes, crops_scores, crops_classes, coords = crops_inference(
        model, img, crops_number, overlap, conf, crops_iou, device, pbar_crops
    )

    # Draw grid if requested
    if draw_grid:
        grid_img = draw_crop_grid(img, coords)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_grid.jpg"), grid_img)
    
    # Stage 3: Smart filtering
    filtered_boxes, filtered_scores, filtered_classes = smart_filter_crops(
        crops_boxes, crops_scores, crops_classes,
        full_boxes, full_scores, full_classes,
        high_iou_threshold=high_iou,
        suspect_iou_threshold=suspect_iou
    )
    
    # Stage 4: Merge
    final_boxes, final_scores, final_classes = merge_detections(
        full_boxes, full_scores, full_classes,
        filtered_boxes, filtered_scores, filtered_classes,
        merge_iou=merge_iou
    )
    
    # Save intermediate results
    if save_intermediate:
        stages = [
            (full_boxes, full_scores, full_classes, "stage1_full"),
            (crops_boxes, crops_scores, crops_classes, "stage2_crops_raw"),
            (filtered_boxes, filtered_scores, filtered_classes, "stage3_crops_filtered")
        ]
        for boxes, scores, classes, suffix in stages:
            if len(boxes) > 0:
                save_annotated_image(img, boxes, scores, classes, model,
                                   out_dir / f"{img_path.stem}_{suffix}.jpg")
    
    # Save final result
    save_annotated_image(img, final_boxes, final_scores, final_classes, model,
                        out_dir / f"{img_path.stem}_hybrid_final.jpg")
    
    return {
        'full': len(full_boxes),
        'crops_raw': len(crops_boxes),
        'crops_filtered': len(filtered_boxes),
        'final': len(final_boxes)
    }


def build_args():
    p = argparse.ArgumentParser(
        description="Hybrid two-stage detection: full + crops with smart filtering"
    )
    p.add_argument("source", help="Input image path or directory")
    p.add_argument("model", help="Path to YOLO weights (.pt)")
    
    p.add_argument("--out_dir", default="hybrid_results", help="Output directory")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--device", default="cuda:0", help="Device (cpu, cuda:0, etc)")
    
    # Crops parameters
    p.add_argument("--crops", type=int, default=6, help="Number of crops (must be even)")
    p.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio for crops [0, 1)")
    p.add_argument("--crops_iou", type=float, default=0.5,
                   help="IoU threshold for WBF in crops inference")
    
    # Filtering parameters
    p.add_argument("--high_iou", type=float, default=0.85,
                   help="High IoU threshold: crops with IoU >= this are validated (keep)")
    p.add_argument("--suspect_iou", type=float, default=0.3,
                   help="Suspect IoU threshold: crops with IoU < this are new objects (keep). Between suspect and high = discard")
    p.add_argument("--merge_iou", type=float, default=0.5,
                   help="IoU threshold for final merging")
    
    # Output options
    p.add_argument("--save_intermediate", action="store_true",
                   help="Save intermediate stage results")
    p.add_argument("--recursive", action="store_true",
                   help="Search images recursively")
    p.add_argument("--draw_grid", action="store_true",
               help="Save original image with crop grid visualization")
    
    return p.parse_args()


def main():
    args = build_args()
    
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    
    if args.crops <= 0 or args.crops % 2 != 0:
        raise SystemExit("❌ --crops must be even and positive")
    
    model = YOLO(args.model)
    images = iter_images(source, args.recursive)
    
    if not images:
        raise SystemExit(f"❌ No images found in: {source}")
    
    print(f"✓ Found {len(images)} image(s)")
    print(f"✓ Pipeline: Full → Crops → Smart Filter → Merge")
    print(f"  Filter logic: keep IoU ≥{args.high_iou} OR IoU <{args.suspect_iou}")
    
    # Progress bars
    pbar_images = tqdm(
        total=len(images),
        desc="Images processed",
        unit="img",
        position=1,
        leave=True,
        bar_format=BAR_FORMAT,
        colour=PURPLE,
        dynamic_ncols=True
    )
    
    pbar_crops = tqdm(
        total=1,
        desc="Current image crops",
        unit="crop",
        position=0,
        leave=False,
        bar_format=BAR_FORMAT,
        colour=GREEN,
        dynamic_ncols=True
    )
    
    try:
        for img_path in images:
            pbar_crops.set_description_str(f"{img_path.name[:24]}")
            stats = process_image_hybrid(
                model=model,
                img_path=img_path,
                out_dir=out_dir,
                conf=args.conf,
                device=args.device,
                crops_number=args.crops,
                overlap=args.overlap,
                crops_iou=args.crops_iou,
                high_iou=args.high_iou,
                suspect_iou=args.suspect_iou,
                merge_iou=args.merge_iou,
                save_intermediate=args.save_intermediate,
                draw_grid=args.draw_grid,
                pbar_crops=pbar_crops
            )
            if stats:
                pbar_images.set_postfix_str(
                    f"Full={stats['full']} | Crops={stats['crops_filtered']} | Final={stats['final']}"
                )
            pbar_images.update(1)
    finally:
        pbar_crops.close()
        pbar_images.close()
    
    print(f"\n✓ Hybrid pipeline complete! Results in {out_dir}")


if __name__ == "__main__":
    main()
