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
        Grid adapts to image orientation: vertical images get more rows than columns.
        
        Args:
            frame: Input image
            crops_number: Number of crops to generate
            
        Returns:
            List of (x_min, y_min, x_max, y_max) tuples
        """
        import math

        height, width = frame.shape[:2]
        is_vertical = height > width
        
        # Base grid dimensions
        base_rows = int(math.sqrt(crops_number))
        base_cols = math.ceil(crops_number / base_rows)
        
        # If vertical image, swap rows/cols to have more rows than columns
        if is_vertical:
            grid_rows = base_cols  # More rows
            grid_cols = base_rows  # Fewer columns
        else:
            grid_rows = base_rows  # Fewer rows
            grid_cols = base_cols  # More columns

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


def weighted_boxes_fusion(boxes, scores, classes, iou_thres: float = 0.5, skip_box_thr: float = 0.0):
    """
    Weighted Boxes Fusion for overlapping detections from sliding windows.
    Uses a clustering approach to merge all overlapping boxes into single detections.
    
    Args:
        boxes: Array of bounding boxes [N, 4] in format (x1, y1, x2, y2)
        scores: Array of confidence scores [N]
        classes: Array of class IDs [N]
        iou_thres: IoU threshold for fusion (boxes with IoU >= threshold are merged)
        skip_box_thr: Minimum confidence score to consider a box
        
    Returns:
        Tuple of (fused_boxes, fused_scores, fused_classes) as numpy arrays
    """
    if len(boxes) == 0:
        return boxes, scores, classes
    
    fused_boxes = []
    fused_scores = []
    fused_classes = []
    
    # Process each class separately
    for cls_id in sorted(set(classes)):
        cls_mask = classes == cls_id
        cls_boxes = boxes[cls_mask].copy()
        cls_scores = scores[cls_mask].copy()
        
        if len(cls_boxes) == 0:
            continue
        
        # Filter by minimum score
        valid_mask = cls_scores >= skip_box_thr
        cls_boxes = cls_boxes[valid_mask]
        cls_scores = cls_scores[valid_mask]
        
        if len(cls_boxes) == 0:
            continue
        
        # Build IoU matrix for clustering
        n = len(cls_boxes)
        iou_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                iou = compute_iou_xyxy(cls_boxes[i], cls_boxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        # Find connected components (clusters of overlapping boxes)
        visited = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if visited[i]:
                continue
            
            # BFS to find all connected boxes
            cluster = [i]
            queue = [i]
            visited[i] = True
            
            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if not visited[j] and iou_matrix[current, j] >= iou_thres:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
            
            # Fuse all boxes in cluster using weighted average
            cluster_boxes = cls_boxes[cluster]
            cluster_scores = cls_scores[cluster]
            
            # Use scores as weights
            weights = cluster_scores / cluster_scores.sum()
            
            # Weighted coordinates
            fused_x1 = np.sum(cluster_boxes[:, 0] * weights)
            fused_y1 = np.sum(cluster_boxes[:, 1] * weights)
            fused_x2 = np.sum(cluster_boxes[:, 2] * weights)
            fused_y2 = np.sum(cluster_boxes[:, 3] * weights)
            
            # Use maximum confidence (more conservative than mean)
            fused_conf = np.max(cluster_scores)
            
            fused_boxes.append([fused_x1, fused_y1, fused_x2, fused_y2])
            fused_scores.append(fused_conf)
            fused_classes.append(cls_id)
    
    if len(fused_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return (
        np.array(fused_boxes, dtype=np.float32),
        np.array(fused_scores, dtype=np.float32),
        np.array(fused_classes, dtype=np.int32)
    )


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


def draw_crop_grid(base_img: np.ndarray, coords, color=(0, 255, 255), thickness=1) -> np.ndarray:
    """
    Draw crop rectangles on a copy of the original image.
    
    Args:
        base_img: Original image
        coords: List of (x_min, y_min, x_max, y_max) crop coordinates (floats)
        color: BGR color for the grid lines
        thickness: Line thickness
        
    Returns:
        Image with crop grid drawn
    """
    grid_img = base_img.copy()
    for (x_min, y_min, x_max, y_max) in coords:
        x1, y1, x2, y2 = map(int, map(round, (x_min, y_min, x_max, y_max)))
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, thickness)
    return grid_img


def process_image(model: YOLO, img_path: Path, out_dir: Path, conf: float, iou: float, device: str,
                  crops_number: int, overlap: float, save_crops: bool, draw_grid: bool,
                  pbar_crops: tqdm | None):
    """
    Process a single image with tiled inference using Weighted Boxes Fusion.
    Crop grid automatically adapts to image orientation.
    
    Args:
        model: YOLO model instance
        img_path: Path to input image
        out_dir: Output directory
        conf: Confidence threshold
        iou: IoU threshold for WBF (boxes with IoU >= threshold are fused)
        device: Device for inference
        crops_number: Number of crops to generate
        overlap: Overlap ratio between crops
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

    # Apply Weighted Boxes Fusion
    if all_boxes:
        boxes = np.array(all_boxes, dtype=np.float32)
        scores = np.array(all_scores, dtype=np.float32)
        classes = np.array(all_classes, dtype=np.int32)

        boxes, scores, classes = weighted_boxes_fusion(
            boxes, scores, classes, 
            iou_thres=iou,
            skip_box_thr=conf
        )

        annotator = Annotator(img, line_width=2, example=model.names)
        for box, score, cls_id in zip(boxes, scores, classes):
            name = model.names.get(int(cls_id), str(int(cls_id)))
            annotator.box_label(box, f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
        img_out = annotator.result()
    else:
        img_out = img

    cv2.imwrite(str(out_dir / f"output_{img_path.stem}_{crops_number}.jpg"), img_out)


def build_args():
    p = argparse.ArgumentParser(description="Tiled inference with Ultralytics YOLO using WBF (Weighted Boxes Fusion).")
    p.add_argument("source", type=str, help="Input image path or directory.")
    p.add_argument("model", type=str, help="Path to YOLO weights (.pt).")

    p.add_argument("--out_dir", type=str, default="aiplocan_wbf_inferences", help="Output directory.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for WBF fusion (recommended 0.5-0.6).")
    p.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cpu, cuda:0).")

    p.add_argument("--crops", type=int, default=4, help="Number of crops (must be even). More than 8 crops is not recommended.")
    p.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio in [0, 1).")
    p.add_argument("--save_crops", action="store_true", help="Save annotated crop images.")
    p.add_argument("--draw_grid", action="store_true", help="Save original image with crop grid drawn.")
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
                draw_grid=args.draw_grid,
                pbar_crops=pbar_crops,
            )
            pbar_images.update(1)
    finally:
        pbar_crops.close()
        pbar_images.close()

    print(f"\n😎 Crops done! Image(s) stored at {args.out_dir}")


if __name__ == "__main__":
    main()
