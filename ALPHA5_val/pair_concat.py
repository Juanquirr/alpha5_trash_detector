import argparse
from pathlib import Path
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def list_images(dir_path: Path):
    return [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def build_index(dir_path: Path, match: str):
    index = {}
    for p in list_images(dir_path):
        key = p.stem if match == "stem" else p.name
        index[key] = p
    return index

def resize_to_same_height(img_a, img_b):
    if img_a is None or img_b is None:
        return None, None
    if img_a.shape[0] == img_b.shape[0]:
        return img_a, img_b
    h = min(img_a.shape[0], img_b.shape[0])
    img_a = cv2.resize(img_a, (int(img_a.shape[1] * h / img_a.shape[0]), h))
    img_b = cv2.resize(img_b, (int(img_b.shape[1] * h / img_b.shape[0]), h))
    return img_a, img_b

def concat_pairs(left_dir: Path, right_dir: Path, out_dir: Path, match: str, suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    left_idx = build_index(left_dir, match)
    right_idx = build_index(right_dir, match)

    keys = sorted(set(left_idx.keys()) & set(right_idx.keys()))
    missing_left = sorted(set(right_idx.keys()) - set(left_idx.keys()))
    missing_right = sorted(set(left_idx.keys()) - set(right_idx.keys()))

    written = 0
    for k in keys:
        left_path = left_idx[k]
        right_path = right_idx[k]

        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        if left_img is None or right_img is None:
            continue

        left_img, right_img = resize_to_same_height(left_img, right_img)
        if left_img is None or right_img is None:
            continue

        out_img = cv2.hconcat([left_img, right_img])
        out_name = f"{left_path.stem}{suffix}.jpg"
        cv2.imwrite(str(out_dir / out_name), out_img)
        written += 1

    print(f"Pairs written: {written}/{len(keys)}")
    print(f"Missing in LEFT dir: {len(missing_left)}")
    print(f"Missing in RIGHT dir: {len(missing_right)}")

def build_args():
    p = argparse.ArgumentParser(description="Concatenate paired images from two folders")
    p.add_argument("left_dir", type=str, help="Directory with left/original images")
    p.add_argument("right_dir", type=str, help="Directory with right/predicted images")
    p.add_argument("--out_dir", type=str, default="concatenated", help="Output directory")
    p.add_argument("--match", type=str, default="name", choices=["name", "stem"],
                   help="How to pair files: full file name or stem (without extension)")
    p.add_argument("--suffix", type=str, default="_concat", help="Suffix for output filenames")
    return p.parse_args()

def main():
    args = build_args()
    concat_pairs(
        left_dir=Path(args.left_dir),
        right_dir=Path(args.right_dir),
        out_dir=Path(args.out_dir),
        match=args.match,
        suffix=args.suffix,
    )

if __name__ == "__main__":
    main()
