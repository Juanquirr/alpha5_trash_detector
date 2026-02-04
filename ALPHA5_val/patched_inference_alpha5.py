import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections

# Progress bar colors
PURPLE = "#8000ff"
GREEN = "#00ff00"

# Custom bar format
BAR_FORMAT = (
    "{desc:<28} "
    "|{bar}| "
    "{percentage:6.2f}% "
    "({n_fmt}/{total_fmt}) "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)


def process_image(
    model: YOLO,
    model_path: str,
    img_path: Path,
    out_dir: Path,
    conf: float,
    iou: float,
    device: str,
    patch_size: int,
    overlap: float,
    nms_threshold: float,
    imgsz: int,
    save_comparison: bool,
    pbar_images: tqdm = None
):
    """
    Process a single image with patched inference.
    
    Args:
        model: YOLO model instance (for class names and baseline comparison)
        model_path: Path to YOLO weights
        img_path: Path to input image
        out_dir: Output directory
        conf: Confidence threshold
        iou: IoU threshold for internal NMS
        device: Device for inference
        patch_size: Size of each patch (square)
        overlap: Overlap ratio between patches [0, 1)
        nms_threshold: IoU threshold for final NMS across patches
        imgsz: Input size for baseline comparison
        save_comparison: Whether to save baseline comparison
        pbar_images: Progress bar for images
    """
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        tqdm.write(f"⚠️  Skipping unreadable image: {img_path}")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if pbar_images is not None:
        pbar_images.set_description_str(f"Processing {img_path.name}")
    
    # Detect with patches
    crops = MakeCropsDetectThem(
        image=img,
        model_path=model_path,
        shape_x=patch_size,
        shape_y=patch_size,
        overlap_x=int(overlap * 100),
        overlap_y=int(overlap * 100),
        conf=conf,
        iou=iou,
        show_processing_status=False
    )
    
    # Combine results with NMS
    result = CombineDetections(crops, nms_threshold=nms_threshold)
    
    # Extract detections
    boxes = result.filtered_boxes
    confidences = result.filtered_confidences
    classes = result.filtered_classes_id
    
    tqdm.write(f"  ✓ {img_path.name}: {len(boxes)} detections (patch_size={patch_size}, overlap={overlap:.0%})")
    
    # Annotate image
    annotator = Annotator(img, line_width=2, example=model.names)
    
    for box, score, cls_id in zip(boxes, confidences, classes):
        box_xyxy = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        name = model.names.get(int(cls_id), str(int(cls_id)))
        annotator.box_label(
            box_xyxy, 
            f"{name} {score:.2f}", 
            color=colors(int(cls_id), bgr=True)
        )
    
    img_out = annotator.result()
    
    # Save result
    output_path = out_dir / f"patched_{img_path.stem}.jpg"
    cv2.imwrite(str(output_path), img_out)
    
    # Optional: baseline comparison
    if save_comparison:
        results_baseline = model.predict(
            img, 
            conf=conf, 
            imgsz=imgsz, 
            verbose=False, 
            device=device
        )
        
        baseline_count = len(results_baseline[0].boxes)
        diff = len(boxes) - baseline_count
        
        tqdm.write(f"  Baseline (imgsz={imgsz}): {baseline_count} | Patched: {len(boxes)} | Diff: {diff:+d}")
        
        # Annotate baseline
        baseline_annotator = Annotator(img, line_width=2, example=model.names)
        for r in results_baseline:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy()
                s = float(box.conf[0].cpu().numpy())
                c = int(box.cls[0].cpu().numpy())
                name = model.names.get(c, str(c))
                baseline_annotator.box_label(
                    [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    f"{name} {s:.2f}",
                    color=colors(c, bgr=True)
                )
        
        baseline_out = baseline_annotator.result()
        baseline_path = out_dir / f"baseline_{img_path.stem}.jpg"
        cv2.imwrite(str(baseline_path), baseline_out)


def iter_images(source: Path, recursive: bool = False):
    """
    Iterate over images in directory or return single image.
    
    Args:
        source: Path to image file or directory
        recursive: Search recursively in subdirectories
    
    Returns:
        List of image paths
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    
    if source.is_file():
        if source.suffix.lower() in supported:
            return [source]
        return []
    
    if recursive:
        images = [p for ext in supported for p in source.rglob(f"*{ext}")]
    else:
        images = [p for ext in supported for p in source.glob(f"*{ext}")]
    
    return sorted(images)


def build_args():
    """Build argument parser"""
    p = argparse.ArgumentParser(
        description="Patched inference with YOLO using patched-yolo-infer library."
    )
    
    # Required arguments
    p.add_argument("source", type=str, 
                   help="Input image path or directory.")
    p.add_argument("model", type=str, 
                   help="Path to YOLO weights (.pt).")
    
    # Output options
    p.add_argument("--out_dir", type=str, default="patched_results",
                   help="Output directory (default: patched_results).")
    
    # Detection parameters
    p.add_argument("--conf", type=float, default=0.3,
                   help="Confidence threshold (default: 0.3).")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for internal NMS (default: 0.5).")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device (e.g., cpu, cuda:0) (default: cuda:0).")
    
    # Patch parameters
    p.add_argument("--patch_size", type=int, default=640,
                   help="Size of each patch in pixels (default: 640).")
    p.add_argument("--overlap", type=float, default=0.25,
                   help="Overlap ratio between patches [0, 1) (default: 0.25).")
    p.add_argument("--nms_threshold", type=float, default=0.25,
                   help="IoU threshold for final NMS across patches (default: 0.25).")
    
    # Comparison options
    p.add_argument("--save_comparison", action="store_true",
                   help="Save baseline comparison images.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input size for baseline comparison (default: 640).")
    
    # Directory options
    p.add_argument("--recursive", action="store_true",
                   help="Search images recursively when source is a directory.")
    
    return p.parse_args()


def main():
    """Main entry point"""
    args = build_args()
    
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    
    # Validate patch size
    if args.patch_size <= 0:
        raise SystemExit("❌ --patch_size must be a positive integer")
    
    # Validate overlap
    if not (0 <= args.overlap < 1):
        raise SystemExit("❌ --overlap must be in range [0, 1)")
    
    # Load model
    model = YOLO(args.model)
    
    # Find images
    images = iter_images(source, recursive=args.recursive)
    
    if not images:
        raise SystemExit(f"❌ No supported images found in: {source}")
    
    # Print configuration
    print(f"✓ Found {len(images)} image(s)")
    print(f"✓ Model: {args.model}")
    print(f"✓ Patch size: {args.patch_size}x{args.patch_size}")
    print(f"✓ Overlap: {args.overlap:.0%}")
    print(f"✓ Confidence: {args.conf}")
    print(f"✓ IoU (internal): {args.iou}")
    print(f"✓ NMS threshold: {args.nms_threshold}")
    print(f"✓ Device: {args.device}")
    
    if args.save_comparison:
        print(f"✓ Baseline comparison enabled (imgsz={args.imgsz})")
    
    # Progress bar
    pbar_images = tqdm(
        total=len(images),
        desc="Images processed",
        unit="img",
        position=0,
        leave=True,
        bar_format=BAR_FORMAT,
        colour=PURPLE,
        dynamic_ncols=True,
    )
    
    try:
        for img_path in images:
            process_image(
                model=model,
                model_path=args.model,
                img_path=img_path,
                out_dir=out_dir,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                patch_size=args.patch_size,
                overlap=args.overlap,
                nms_threshold=args.nms_threshold,
                imgsz=args.imgsz,
                save_comparison=args.save_comparison,
                pbar_images=pbar_images
            )
            pbar_images.update(1)
    
    finally:
        pbar_images.close()
    
    print(f"\n✓ Patched inference complete! Results in {out_dir}")


if __name__ == '__main__':
    main()
