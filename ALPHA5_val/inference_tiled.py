"""
inference_tiled.py

Tiled inference with Ultralytics YOLO supporting both NMS and WBF fusion methods.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

from crop_utils import UniformCrops, draw_crop_grid, iter_images
from wbf_utils import compute_iou_xyxy, weighted_boxes_fusion, greedy_nms_classwise


PURPLE = "#8000ff"
GREEN = "#00ff00"
BAR_FORMAT = (
    "{desc:<28} "
    "|{bar}| "
    "{percentage:6.2f}% "
    "({n_fmt}/{total_fmt}) "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)


def process_image(model: YOLO, img_path: Path, out_dir: Path, conf: float, iou: float, 
                 device: str, crops_number: int, overlap: float, fusion_method: str,
                 save_crops: bool, draw_grid: bool, pbar_crops: tqdm = None):
    """
    Process a single image with tiled inference.
    Supports both NMS and WBF fusion methods.
    
    Args:
        model: YOLO model instance
        img_path: Path to input image
        out_dir: Output directory
        conf: Confidence threshold
        iou: IoU threshold for fusion method
        device: Device for inference
        crops_number: Number of crops to generate
        overlap: Overlap ratio between crops
        fusion_method: 'nms' or 'wbf'
        save_crops: Whether to save individual crop predictions
        draw_grid: Whether to save the original image with crop grid drawn
        pbar_crops: Progress bar for crops
    """
    img = cv2.imread(str(img_path))
    if img is None:
        tqdm.write(f"⚠️  Skipping unreadable image: {img_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / f"{img_path.stem}_crops_{crops_number}"
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    cropper = UniformCrops(overlap_ratio=overlap)
    crops, coords = cropper.crop(img, crops_number=crops_number)

    # Optionally draw crop grid on original image
    if draw_grid:
        grid_img = draw_crop_grid(img, coords, color=(0, 255, 255), thickness=1)
        grid_path = out_dir / f"{img_path.stem}_grid_{crops_number}.jpg"
        cv2.imwrite(str(grid_path), grid_img)

    all_boxes, all_scores, all_classes = [], [], []

    if pbar_crops is not None:
        pbar_crops.reset(total=len(crops))
        pbar_crops.set_description_str(f"{img_path.name}")

    for idx, (crop, (x_min, y_min, _, _)) in enumerate(zip(crops, coords)):
        results = model.predict(crop, device=device, conf=conf, verbose=False)
        r = results[0]

        if save_crops:
            crop_annot = r.plot()
            cv2.imwrite(str(crops_dir / f"crop_{idx:02d}.jpg"), crop_annot)

        if r.boxes is None or len(r.boxes) == 0:
            if pbar_crops is not None:
                pbar_crops.update(1)
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

        if pbar_crops is not None:
            pbar_crops.update(1)

    # Apply fusion method
    if all_boxes:
        boxes = np.array(all_boxes, dtype=np.float32)
        scores = np.array(all_scores, dtype=np.float32)
        classes = np.array(all_classes, dtype=np.int32)

        if fusion_method == "wbf":
            # Weighted Boxes Fusion
            boxes, scores, classes = weighted_boxes_fusion(
                boxes, scores, classes,
                iou_thres=iou,
                skip_box_thr=conf
            )
        else:  # nms
            # Classic Non-Maximum Suppression
            keep = greedy_nms_classwise(boxes, scores, classes, iou_thres=iou)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        annotator = Annotator(img, line_width=2, example=model.names)
        for box, score, cls_id in zip(boxes, scores, classes):
            name = model.names.get(int(cls_id), str(int(cls_id)))
            annotator.box_label(box, f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
        img_out = annotator.result()
    else:
        img_out = img

    cv2.imwrite(str(out_dir / f"output_{img_path.stem}_{crops_number}.jpg"), img_out)


def build_args():
    p = argparse.ArgumentParser(
        description="Tiled inference with Ultralytics YOLO supporting NMS and WBF fusion methods."
    )
    p.add_argument("source", type=str, help="Input image path or directory.")
    p.add_argument("model", type=str, help="Path to YOLO weights (.pt).")
    
    p.add_argument("--out_dir", type=str, default="tiled_inferences", 
                   help="Output directory.")
    p.add_argument("--conf", type=float, default=0.25, 
                   help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5, 
                   help="IoU threshold for fusion (0.45 for NMS, 0.5-0.6 for WBF).")
    p.add_argument("--device", type=str, default="cuda:0", 
                   help="Device (e.g., cpu, cuda:0).")
    
    p.add_argument("--crops", type=int, default=4, 
                   help="Number of crops (must be even). Max 8 recommended.")
    p.add_argument("--overlap", type=float, default=0.2, 
                   help="Overlap ratio in [0, 1).")
    
    p.add_argument("--fusion", choices=["nms", "wbf"], default="wbf",
                   help="Fusion method: 'nms' (classic suppression) or 'wbf' (weighted fusion).")
    
    p.add_argument("--save_crops", action="store_true", 
                   help="Save annotated crop images.")
    p.add_argument("--draw_grid", action="store_true", 
                   help="Save original image with crop grid drawn.")
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
    print(f"✓ Fusion method: {args.fusion.upper()}")
    
    pbar_images = tqdm(
        total=len(images),
        desc="Images processed",
        unit="img",
        position=1,
        leave=True,
        bar_format=BAR_FORMAT,
        colour=PURPLE,
        dynamic_ncols=True,
    )
    
    pbar_crops = tqdm(
        total=1,
        desc="Waiting...",
        unit="crop",
        position=0,
        leave=False,
        bar_format=BAR_FORMAT,
        colour=GREEN,
        dynamic_ncols=True,
    )
    
    try:
        for img_path in images:
            process_image(
                model=model,
                img_path=img_path,
                out_dir=out_dir,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                crops_number=args.crops,
                overlap=args.overlap,
                fusion_method=args.fusion,
                save_crops=args.save_crops,
                draw_grid=args.draw_grid,
                pbar_crops=pbar_crops,
            )
            pbar_images.update(1)
    finally:
        pbar_crops.close()
        pbar_images.close()
    
    print(f"\n✓ Tiled inference complete! Results in {args.out_dir}")


if __name__ == "__main__":
    main()
