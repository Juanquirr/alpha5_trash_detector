import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png"}
PURPLE = "#8000ff"


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


def draw_crop_grid(base_img: np.ndarray, coords, color=(0, 255, 255), thickness=2) -> np.ndarray:
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
    for idx, (x_min, y_min, x_max, y_max) in enumerate(coords):
        x1, y1, x2, y2 = map(int, map(round, (x_min, y_min, x_max, y_max)))
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, thickness)
        
        # Optional: draw crop number
        label_pos = (x1 + 5, y1 + 20)
        cv2.putText(grid_img, f"#{idx}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, thickness)
    
    return grid_img


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


def process_image(img_path: Path, out_dir: Path, crops_number: int, overlap: float, 
                  draw_grid: bool, save_crops: bool):
    """
    Extract crops from a single image.
    
    Args:
        img_path: Path to input image
        out_dir: Output directory
        crops_number: Number of crops to generate
        overlap: Overlap ratio between crops
        draw_grid: Whether to save the original image with crop grid drawn
        save_crops: Whether to save individual crop images
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Skipping unreadable image: {img_path}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cropper
    cropper = UniformCrops(overlap_ratio=overlap)
    crops, coords = cropper.crop(img, crops_number=crops_number)
    
    # Save grid visualization
    if draw_grid:
        grid_img = draw_crop_grid(img, coords, color=(0, 255, 255), thickness=2)
        grid_path = out_dir / f"{img_path.stem}_grid_{crops_number}.jpg"
        cv2.imwrite(str(grid_path), grid_img)
    
    # Save individual crops
    saved = 0
    if save_crops:
        crops_dir = out_dir / f"{img_path.stem}_crops_{crops_number}"
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, crop in enumerate(crops):
            crop_path = crops_dir / f"crop_{idx:02d}.jpg"
            if cv2.imwrite(str(crop_path), crop):
                saved += 1
    
    return saved


def build_args():
    p = argparse.ArgumentParser(
        description="Extract uniform overlapping crops from images without inference."
    )
    p.add_argument("source", type=str, help="Input image path or directory.")
    
    p.add_argument("--out_dir", type=str, default="image_crops", help="Output directory.")
    p.add_argument("--crops", type=int, default=4, help="Number of crops (must be even).")
    p.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio in [0, 1).")
    p.add_argument("--save_crops", action="store_true", help="Save individual crop images.")
    p.add_argument("--draw_grid", action="store_true", help="Save original image with crop grid drawn.")
    p.add_argument("--recursive", action="store_true", help="Search images recursively when source is a directory.")
    
    return p.parse_args()


def main():
    args = build_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)

    if args.crops <= 0 or args.crops % 2 != 0:
        raise SystemExit("❌ --crops must be an even positive integer")

    images = iter_images(source, recursive=args.recursive)
    if not images:
        raise SystemExit(f"❌ No supported images found in: {source}")

    print(f"✓ Found {len(images)} image(s)")
    print(f"✓ Extracting {args.crops} crops per image (overlap={args.overlap})")

    total_crops_saved = 0
    
    for img_path in tqdm(images, desc="Processing images", unit="img", colour=PURPLE):
        crops_saved = process_image(
            img_path=img_path,
            out_dir=out_dir,
            crops_number=args.crops,
            overlap=args.overlap,
            draw_grid=args.draw_grid,
            save_crops=args.save_crops,
        )
        total_crops_saved += crops_saved

    print(f"\n✓ Done!")
    print(f"  Images processed: {len(images)}")
    if args.save_crops:
        print(f"  Total crops saved: {total_crops_saved}")
    if args.draw_grid:
        print(f"  Grid visualizations: {len(images)}")
    print(f"  Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
