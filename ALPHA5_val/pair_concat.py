import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(dir_path: Path, recursive: bool = True) -> list[Path]:
    """
    List all image files in a directory, sorted alphabetically.
    
    Args:
        dir_path: Directory to search
        recursive: If True, search recursively in subdirectories
        
    Returns:
        Sorted list of image file paths
    """
    if not dir_path.exists():
        return []
    
    if recursive:
        images = [p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        images = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    
    return sorted(images)


def resize_to_same_height(img_a, img_b):
    """
    Resize two images to have the same height while preserving aspect ratio.
    
    Args:
        img_a: First image (numpy array)
        img_b: Second image (numpy array)
        
    Returns:
        Tuple of (resized_img_a, resized_img_b), or (None, None) if inputs are invalid
    """
    if img_a is None or img_b is None:
        return None, None
    if img_a.shape[0] == img_b.shape[0]:
        return img_a, img_b
    
    h = min(img_a.shape[0], img_b.shape[0])
    new_w_a = int(img_a.shape[1] * h / img_a.shape[0])
    new_w_b = int(img_b.shape[1] * h / img_b.shape[0])
    
    img_a = cv2.resize(img_a, (new_w_a, h))
    img_b = cv2.resize(img_b, (new_w_b, h))
    return img_a, img_b


def concat_pairs_by_order(left_dir: Path, right_dir: Path, out_dir: Path, suffix: str, recursive: bool):
    """
    Concatenate images from two directories side-by-side by alphabetical order.
    Assumes both directories have the same number of images in corresponding order.
    
    Args:
        left_dir: Directory with left/original images
        right_dir: Directory with right/predicted images
        out_dir: Output directory for concatenated images
        suffix: Suffix to append to output filenames
        recursive: Whether to search recursively
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Scanning directories...")
    left_images = list_images(left_dir, recursive=recursive)
    right_images = list_images(right_dir, recursive=recursive)

    if not left_images:
        raise SystemExit(f"❌ No images found in left directory: {left_dir}")
    if not right_images:
        raise SystemExit(f"❌ No images found in right directory: {right_dir}")

    print(f"✓ Left dir: {len(left_images)} images")
    print(f"✓ Right dir: {len(right_images)} images")

    if len(left_images) != len(right_images):
        print(f"⚠️  Warning: Different number of images in each directory!")
        print(f"   Will process {min(len(left_images), len(right_images))} pairs")

    n_pairs = min(len(left_images), len(right_images))

    written = 0
    skipped = 0

    for i in tqdm(range(n_pairs), desc="Concatenating pairs", unit="pair", colour="#8000ff"):
        left_path = left_images[i]
        right_path = right_images[i]

        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        
        if left_img is None:
            tqdm.write(f"⚠️  Cannot read left: {left_path.name}")
            skipped += 1
            continue
        
        if right_img is None:
            tqdm.write(f"⚠️  Cannot read right: {right_path.name}")
            skipped += 1
            continue

        left_img, right_img = resize_to_same_height(left_img, right_img)
        if left_img is None or right_img is None:
            tqdm.write(f"⚠️  Resize failed for pair {i+1}")
            skipped += 1
            continue

        out_img = cv2.hconcat([left_img, right_img])
        out_name = f"{left_path.stem}{suffix}.jpg"
        out_path = out_dir / out_name
        
        success = cv2.imwrite(str(out_path), out_img)
        if not success:
            tqdm.write(f"⚠️  Failed to write: {out_path.name}")
            skipped += 1
        else:
            written += 1

    print(f"\n✓ Concatenation complete!")
    print(f"  Pairs written: {written}/{n_pairs}")
    if skipped > 0:
        print(f"  Skipped: {skipped}")


def build_args():
    p = argparse.ArgumentParser(
        description="Concatenate paired images from two folders by alphabetical order (no matching required)"
    )
    p.add_argument("left_dir", type=str, help="Directory with left/original images")
    p.add_argument("right_dir", type=str, help="Directory with right/predicted images")
    p.add_argument("--out_dir", type=str, default="concatenated", help="Output directory")
    p.add_argument("--suffix", type=str, default="_concat", help="Suffix for output filenames")
    p.add_argument("--no-recursive", action="store_true", help="Don't search recursively in subdirectories")
    return p.parse_args()


def main():
    args = build_args()
    
    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    
    if not left_dir.exists():
        raise SystemExit(f"❌ Left directory not found: {left_dir}")
    if not right_dir.exists():
        raise SystemExit(f"❌ Right directory not found: {right_dir}")
    
    concat_pairs_by_order(
        left_dir=left_dir,
        right_dir=right_dir,
        out_dir=Path(args.out_dir),
        suffix=args.suffix,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
