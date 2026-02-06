import argparse
from pathlib import Path
from typing import List

# Supported image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}


def find_images_without_labels(dataset_path: Path) -> List[Path]:
    """
    Find images that don't have corresponding label files.
    
    Args:
        dataset_path: Path to dataset split (e.g., train, val, test)
    
    Returns:
        List of image paths missing label files
    """
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not labels_dir.exists():
        print(f"⚠️  Labels directory not found, creating: {labels_dir}")
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = [
        f for f in images_dir.iterdir() 
        if f.is_file() and f.suffix in IMAGE_EXTS
    ]
    
    # Check which ones are missing labels
    missing = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing.append(img_path)
    
    return missing


def create_empty_labels(image_paths: List[Path], labels_dir: Path, dry_run: bool = False):
    """
    Create empty label files for given images.
    
    Args:
        image_paths: List of image paths needing empty labels
        labels_dir: Directory where labels should be created
        dry_run: If True, only print what would be created
    """
    if not image_paths:
        print("✓ All images have corresponding label files")
        return
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Found {len(image_paths)} images without labels:")
    
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        print(f"  - {img_path.name} → {label_path.name}")
        
        if not dry_run:
            # Create empty file
            label_path.touch()
    
    if not dry_run:
        print(f"\n✓ Created {len(image_paths)} empty label files")
    else:
        print(f"\n[DRY RUN] Would create {len(image_paths)} empty label files")


def check_dataset_structure(dataset_path: Path):
    """Print dataset structure summary"""
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        return
    
    image_files = [
        f for f in images_dir.iterdir() 
        if f.is_file() and f.suffix in IMAGE_EXTS
    ]
    
    label_files = []
    if labels_dir.exists():
        label_files = [f for f in labels_dir.iterdir() if f.suffix == '.txt']
    
    # Count empty vs non-empty labels
    empty_labels = 0
    non_empty_labels = 0
    
    for label_file in label_files:
        if label_file.stat().st_size == 0:
            empty_labels += 1
        else:
            non_empty_labels += 1
    
    print(f"\nDataset: {dataset_path.name}")
    print(f"  Images:             {len(image_files)}")
    print(f"  Label files:        {len(label_files)}")
    print(f"    - Empty (neg):    {empty_labels}")
    print(f"    - Non-empty (pos): {non_empty_labels}")
    print(f"  Missing labels:     {len(image_files) - len(label_files)}")


def build_args():
    """Build argument parser"""
    p = argparse.ArgumentParser(
        description="Create empty label files for negative samples in YOLO dataset."
    )
    
    p.add_argument("dataset_path", type=str,
                   help="Path to dataset split (e.g., train, val, test).")
    
    p.add_argument("--dry_run", action="store_true",
                   help="Show what would be created without creating files.")
    
    p.add_argument("--check_only", action="store_true",
                   help="Only check and report dataset structure.")
    
    return p.parse_args()


def main():
    """Main entry point"""
    args = build_args()
    
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        raise SystemExit(f"❌ Dataset path not found: {dataset_path}")
    
    # Check structure
    print("="*60)
    print("DATASET STRUCTURE CHECK")
    print("="*60)
    check_dataset_structure(dataset_path)
    
    if args.check_only:
        return
    
    # Find missing labels
    print("\n" + "="*60)
    print("SEARCHING FOR IMAGES WITHOUT LABELS")
    print("="*60)
    
    missing = find_images_without_labels(dataset_path)
    labels_dir = dataset_path / "labels"
    
    # Create empty labels
    create_empty_labels(missing, labels_dir, dry_run=args.dry_run)
    
    # Final check
    if not args.dry_run and missing:
        print("\n" + "="*60)
        print("FINAL STRUCTURE")
        print("="*60)
        check_dataset_structure(dataset_path)


if __name__ == '__main__':
    main()

