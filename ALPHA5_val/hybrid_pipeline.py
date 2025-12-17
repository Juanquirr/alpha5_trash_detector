import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

# Import from your existing scripts
from static_slices import (
    UniformCrops,
    compute_iou_xyxy,
    weighted_boxes_fusion,
    iter_images,
    draw_crop_grid,
    IMG_EXTS,
    PURPLE,
    GREEN,
    BAR_FORMAT
)


def full_image_inference(model: YOLO, img: np.ndarray, conf: float, device: str):
    """
    Run inference on full image without crops.
    
    Args:
        model: YOLO model instance
        img: Input image (numpy array)
        conf: Confidence threshold
        device: Device for inference
        
    Returns:
        Tuple of (boxes, scores, classes) as numpy arrays
    """
    results = model.predict(img, device=device, conf=conf, verbose=False)
    r = results[0]
    
    if r.boxes is None or len(r.boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    
    return boxes, scores, classes


def crops_inference(model: YOLO, img: np.ndarray, crops_number: int, overlap: float,
                   conf: float, iou: float, device: str):
    """
    Run inference using crops with WBF fusion.
    
    Args:
        model: YOLO model instance
        img: Input image (numpy array)
        crops_number: Number of crops to generate
        overlap: Overlap ratio between crops
        conf: Confidence threshold
        iou: IoU threshold for WBF fusion
        device: Device for inference
        
    Returns:
        Tuple of (boxes, scores, classes) as numpy arrays
    """
    cropper = UniformCrops(overlap_ratio=overlap)
    crops, coords = cropper.crop(img, crops_number=crops_number)
    
    all_boxes, all_scores, all_classes = [], [], []
    
    for crop, (x_min, y_min, _, _) in zip(crops, coords):
        results = model.predict(crop, device=device, conf=conf, verbose=False)
        r = results[0]
        
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
        return np.array([]), np.array([]), np.array([])
    
    boxes = np.array(all_boxes, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    classes = np.array(all_classes, dtype=np.int32)
    
    # Apply WBF
    boxes, scores, classes = weighted_boxes_fusion(
        boxes, scores, classes,
        iou_thres=iou,
        skip_box_thr=conf
    )
    
    return boxes, scores, classes


def filter_crops_by_full(crops_boxes, crops_scores, crops_classes,
                        full_boxes, full_scores, full_classes,
                        iou_threshold: float = 0.9):
    """
    Filter crops detections: keep only those with high IoU with full-image detections.
    This removes spurious detections that only appear in crops.
    
    Args:
        crops_boxes: Boxes from crops inference [N, 4]
        crops_scores: Scores from crops inference [N]
        crops_classes: Classes from crops inference [N]
        full_boxes: Boxes from full-image inference [M, 4]
        full_scores: Scores from full-image inference [M]
        full_classes: Classes from full-image inference [M]
        iou_threshold: Minimum IoU to consider a crops detection valid
        
    Returns:
        Tuple of filtered (boxes, scores, classes)
    """
    if len(crops_boxes) == 0:
        return crops_boxes, crops_scores, crops_classes
    
    if len(full_boxes) == 0:
        # No full detections, return all crops (fallback mode)
        return crops_boxes, crops_scores, crops_classes
    
    valid_indices = []
    
    for i, (crop_box, crop_cls) in enumerate(zip(crops_boxes, crops_classes)):
        # Check if this crops detection has high overlap with any full detection
        # of the same class
        has_support = False
        
        for full_box, full_cls in zip(full_boxes, full_classes):
            # Only compare same-class detections
            if crop_cls != full_cls:
                continue
            
            iou = compute_iou_xyxy(crop_box, full_box)
            
            if iou >= iou_threshold:
                has_support = True
                break
        
        if has_support:
            valid_indices.append(i)
    
    if not valid_indices:
        return np.array([]), np.array([]), np.array([])
    
    return (
        crops_boxes[valid_indices],
        crops_scores[valid_indices],
        crops_classes[valid_indices]
    )


def merge_detections(full_boxes, full_scores, full_classes,
                    crops_boxes, crops_scores, crops_classes,
                    merge_iou: float = 0.5):
    """
    Merge full and filtered crops detections using WBF to avoid duplicates.
    
    Args:
        full_boxes, full_scores, full_classes: Full-image detections
        crops_boxes, crops_scores, crops_classes: Filtered crops detections
        merge_iou: IoU threshold for merging duplicates
        
    Returns:
        Tuple of merged (boxes, scores, classes)
    """
    # Concatenate all detections
    all_boxes = np.concatenate([full_boxes, crops_boxes], axis=0) if len(crops_boxes) > 0 else full_boxes
    all_scores = np.concatenate([full_scores, crops_scores], axis=0) if len(crops_scores) > 0 else full_scores
    all_classes = np.concatenate([full_classes, crops_classes], axis=0) if len(crops_classes) > 0 else full_classes
    
    if len(all_boxes) == 0:
        return all_boxes, all_scores, all_classes
    
    # Final WBF to merge overlapping detections from both sources
    return weighted_boxes_fusion(
        all_boxes, all_scores, all_classes,
        iou_thres=merge_iou,
        skip_box_thr=0.0
    )


def process_image_hybrid(model: YOLO, img_path: Path, out_dir: Path,
                        conf: float, device: str,
                        crops_number: int, overlap: float, crops_iou: float,
                        filter_iou: float, merge_iou: float,
                        save_intermediate: bool):
    """
    Process image using hybrid two-stage pipeline.
    
    Args:
        model: YOLO model instance
        img_path: Path to input image
        out_dir: Output directory
        conf: Confidence threshold
        device: Device for inference
        crops_number: Number of crops for stage 2
        overlap: Overlap ratio for crops
        crops_iou: IoU threshold for WBF in crops inference
        filter_iou: IoU threshold for filtering crops vs full detections
        merge_iou: IoU threshold for merging full + crops
        save_intermediate: Save intermediate results (full-only, crops-only)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Skipping unreadable image: {img_path}")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Full-image inference
    full_boxes, full_scores, full_classes = full_image_inference(
        model, img, conf, device
    )
    
    # Stage 2: Crops inference
    crops_boxes, crops_scores, crops_classes = crops_inference(
        model, img, crops_number, overlap, conf, crops_iou, device
    )
    
    # Stage 3: Filter crops by full detections
    filtered_boxes, filtered_scores, filtered_classes = filter_crops_by_full(
        crops_boxes, crops_scores, crops_classes,
        full_boxes, full_scores, full_classes,
        iou_threshold=filter_iou
    )
    
    # Stage 4: Merge full + filtered crops
    final_boxes, final_scores, final_classes = merge_detections(
        full_boxes, full_scores, full_classes,
        filtered_boxes, filtered_scores, filtered_classes,
        merge_iou=merge_iou
    )
    
    # Save intermediate results if requested
    if save_intermediate:
        # Full-only
        if len(full_boxes) > 0:
            img_full = img.copy()
            annotator = Annotator(img_full, line_width=2, example=model.names)
            for box, score, cls_id in zip(full_boxes, full_scores, full_classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(box, f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
            cv2.imwrite(str(out_dir / f"{img_path.stem}_stage1_full.jpg"), annotator.result())
        
        # Crops-only (before filtering)
        if len(crops_boxes) > 0:
            img_crops = img.copy()
            annotator = Annotator(img_crops, line_width=2, example=model.names)
            for box, score, cls_id in zip(crops_boxes, crops_scores, crops_classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(box, f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
            cv2.imwrite(str(out_dir / f"{img_path.stem}_stage2_crops_raw.jpg"), annotator.result())
        
        # Filtered crops
        if len(filtered_boxes) > 0:
            img_filt = img.copy()
            annotator = Annotator(img_filt, line_width=2, example=model.names)
            for box, score, cls_id in zip(filtered_boxes, filtered_scores, filtered_classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(box, f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
            cv2.imwrite(str(out_dir / f"{img_path.stem}_stage3_crops_filtered.jpg"), annotator.result())
    
    # Save final merged result
    img_out = img.copy()
    if len(final_boxes) > 0:
        annotator = Annotator(img_out, line_width=2, example=model.names)
        for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
            name = model.names.get(int(cls_id), str(int(cls_id)))
            annotator.box_label(box, f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
        img_out = annotator.result()
    
    cv2.imwrite(str(out_dir / f"{img_path.stem}_hybrid_final.jpg"), img_out)
    
    # Print stats
    print(f"\n📊 {img_path.name}:")
    print(f"   Full detections: {len(full_boxes)}")
    print(f"   Crops detections (raw): {len(crops_boxes)}")
    print(f"   Crops detections (filtered): {len(filtered_boxes)}")
    print(f"   Final merged: {len(final_boxes)}")


def build_args():
    p = argparse.ArgumentParser(
        description="Hybrid two-stage detection: full + crops with IoU-based filtering"
    )
    p.add_argument("source", type=str, help="Input image path or directory.")
    p.add_argument("model", type=str, help="Path to YOLO weights (.pt).")
    
    p.add_argument("--out_dir", type=str, default="hybrid_results", help="Output directory.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cpu, cuda:0).")
    
    # Crops parameters
    p.add_argument("--crops", type=int, default=6, help="Number of crops (must be even).")
    p.add_argument("--overlap", type=float, default=0.4, help="Overlap ratio for crops in [0, 1).")
    p.add_argument("--crops_iou", type=float, default=0.5, help="IoU threshold for WBF in crops inference.")
    
    # Filtering parameters
    p.add_argument("--filter_iou", type=float, default=0.9,
                   help="Minimum IoU between crops and full detections to keep crops detection (0.85-0.95 recommended).")
    p.add_argument("--merge_iou", type=float, default=0.5,
                   help="IoU threshold for final merging of full + filtered crops.")
    
    # Output options
    p.add_argument("--save_intermediate", action="store_true",
                   help="Save intermediate results (full-only, crops-raw, crops-filtered).")
    p.add_argument("--recursive", action="store_true",
                   help="Search images recursively when source is a directory.")
    
    return p.parse_args()


def main():
    args = build_args()
    
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    
    if args.crops <= 0 or args.crops % 2 != 0:
        raise SystemExit("❌ --crops must be an even positive integer")
    
    model = YOLO(args.model)
    
    images = iter_images(source, recursive=args.recursive)
    if not images:
        raise SystemExit(f"❌ No supported images found in: {source}")
    
    print(f"✓ Found {len(images)} image(s)")
    print(f"✓ Pipeline: Full → Crops → Filter (IoU≥{args.filter_iou}) → Merge")
    
    for img_path in tqdm(images, desc="Processing images", unit="img", colour=PURPLE):
        process_image_hybrid(
            model=model,
            img_path=img_path,
            out_dir=out_dir,
            conf=args.conf,
            device=args.device,
            crops_number=args.crops,
            overlap=args.overlap,
            crops_iou=args.crops_iou,
            filter_iou=args.filter_iou,
            merge_iou=args.merge_iou,
            save_intermediate=args.save_intermediate,
        )
    
    print(f"\n✓ Hybrid pipeline complete! Results in {args.out_dir}")


if __name__ == "__main__":
    main()
