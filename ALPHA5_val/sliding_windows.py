import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

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

class UniformCrops:
    """Uniform overlapping crops for a frame."""
    def __init__(self, overlap_ratio: float) -> None:
        if not (0 <= overlap_ratio < 1):
            raise ValueError("overlap_ratio must be in [0, 1)")
        self._overlap_ratio = overlap_ratio

    def crop(self, frame: np.ndarray, crops_number: int):
        """
        Split frame into uniform overlapping crops.
        
        Args:
            frame: Input image (numpy array)
            crops_number: Number of crops (must be even and positive)
            
        Returns:
            Tuple of (crops_list, coordinates_list)
        """
        if (crops_number % 2 != 0) or (crops_number <= 0):
            raise ValueError("crops_number must be even and positive")

        coords = self._get_crops_coords(frame, crops_number)
        crops = []
        for x_min, y_min, x_max, y_max in coords:
            x1, y1, x2, y2 = map(int, map(round, (x_min, y_min, x_max, y_max)))
            crops.append(frame[y1:y2, x1:x2])
        return crops, coords

    def _get_crops_coords(self, frame: np.ndarray, crops_number: int):
        """
        Calculate crop coordinates for uniform grid with overlap.
        
        Args:
            frame: Input image
            crops_number: Number of crops to generate
            
        Returns:
            List of (x_min, y_min, x_max, y_max) tuples
        """
        import math

        height, width = frame.shape[:2]
        grid_rows = int(math.sqrt(crops_number))
        grid_cols = math.ceil(crops_number / grid_rows)

        cell_w = width / (grid_cols - (grid_cols - 1) * self._overlap_ratio)
        cell_h = height / (grid_rows - (grid_rows - 1) * self._overlap_ratio)

        stride_w = cell_w * (1 - self._overlap_ratio)
        stride_h = cell_h * (1 - self._overlap_ratio)

        coords = []
        count = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                if count >= crops_number:
                    break

                x_min = c * stride_w
                y_min = r * stride_h
                x_max = x_min + cell_w
                y_max = y_min + cell_h

                x_min = max(0.0, x_min)
                y_min = max(0.0, y_min)
                x_max = min(float(width), x_max)
                y_max = min(float(height), y_max)

                coords.append((x_min, y_min, x_max, y_max))
                count += 1

        return coords

def compute_iou_xyxy(a, b) -> float:
    """
    Compute Intersection over Union for two bounding boxes in xyxy format.
    
    Args:
        a: First box [x1, y1, x2, y2]
        b: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def greedy_nms_classwise(boxes, scores, classes, iou_thres: float):
    """
    Apply class-wise Non-Maximum Suppression.
    
    Args:
        boxes: Array of bounding boxes
        scores: Array of confidence scores
        classes: Array of class IDs
        iou_thres: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    keep = []
    for cls_id in sorted(set(classes)):
        idxs = [i for i, c in enumerate(classes) if c == cls_id]
        idxs.sort(key=lambda i: scores[i], reverse=True)

        picked = []
        for i in idxs:
            ok = True
            for j in picked:
                if compute_iou_xyxy(boxes[i], boxes[j]) > iou_thres:
                    ok = False
                    break
            if ok:
                picked.append(i)

        keep.extend(picked)
    keep.sort(key=lambda i: scores[i], reverse=True)
    return keep

def iter_images(source: Path, recursive: bool):
    """
    List all image files from source path.
    
    Args:
        source: File or directory path
        recursive: Search recursively if True
        
    Returns:
        List of image file paths
    """
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXTS else []
    if source.is_dir():
        it = source.rglob("*") if recursive else source.iterdir()
        return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS])
    return []

def process_image(model: YOLO, img_path: Path, out_dir: Path, conf: float, iou: float, device: str,
                  crops_number: int, overlap: float, save_crops: bool, pbar_crops: tqdm | None):
    """
    Process a single image with tiled inference.
    
    Args:
        model: YOLO model instance
        img_path: Path to input image
        out_dir: Output directory
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: Device for inference
        crops_number: Number of crops to generate
        overlap: Overlap ratio between crops
        save_crops: Whether to save individual crop predictions
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

    if all_boxes:
        boxes = np.array(all_boxes, dtype=np.float32)
        scores = np.array(all_scores, dtype=np.float32)
        classes = np.array(all_classes, dtype=np.int32)

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
    p = argparse.ArgumentParser(description="Tiled inference with Ultralytics YOLO using a custom uniform cropper.")
    p.add_argument("source", type=str, help="Input image path or directory.")
    p.add_argument("model", type=str, help="Path to YOLO weights (.pt).")

    p.add_argument("--out_dir", type=str, default="aiplocan_tiled_inferences", help="Output directory.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for global NMS.")
    p.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cpu, cuda:0).")

    p.add_argument("--crops", type=int, default=4, help="Number of crops (must be even). More than 8 crops is not recomended.")
    p.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio in [0, 1).")
    p.add_argument("--save_crops", action="store_true", help="Save annotated crop images.")
    p.add_argument("--recursive", action="store_true", help="Search images recursively when source is a directory.")
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
                save_crops=args.save_crops,
                pbar_crops=pbar_crops,
            )
            pbar_images.update(1)
    finally:
        pbar_crops.close()
        pbar_images.close()

    print(f"\n😎 Crops done! Image(s) stored at {args.out_dir}")

if __name__ == "__main__":
    main()
