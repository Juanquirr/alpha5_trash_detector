import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

IMG_EXTS = {".jpg", ".jpeg", ".png"}

PURPLE = "#8000ff"
BAR_FORMAT = (
    "{desc:<28} "
    "|{bar}| "
    "{percentage:6.2f}% "
    "({n_fmt}/{total_fmt}) "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAHI sliced inference (Ultralytics) with dual tqdm bars.")

    p.add_argument("source", type=str, help="Input image file or directory.")
    p.add_argument("model_path", type=str, help="Path to YOLO .pt weights.")
    p.add_argument("project", type=str, help="Output directory.")

    p.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cpu, cuda:0).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    
    p.add_argument("--slice_height", type=int, default=320, help="Slice height.")
    p.add_argument("--slice_width", type=int, default=320, help="Slice width.")
    p.add_argument("--overlap_height_ratio", type=float, default=0.2, help="Slice overlap height ratio.")
    p.add_argument("--overlap_width_ratio", type=float, default=0.2, help="Slice overlap width ratio.")

    p.add_argument("--recursive", action="store_true", help="Recursively search images if source is a directory.")
    p.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format.",
    )
    return p.parse_args()


def list_images(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXTS else []
    if source.is_dir():
        it = source.rglob("*") if recursive else source.iterdir()
        return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS])
    return []


def safe_unique_name(img_path: Path) -> str:
    """
    Build a unique stem from the full relative path to avoid collisions
    (e.g., sub/a.jpg and other/a.jpg). No extension.
    """
    return img_path.as_posix().replace("/", "__").replace("\\", "__").rsplit(".", 1)[0]


def export_visual(result, out_dir: Path, file_stem: str, fmt: str):
    """
    Wrapper around PredictionResult.export_visuals compatible with
    SAHI versions with and without `export_format`.
    Saves exactly one file per image with a unique name.
    """
    fmt = fmt.lower().lstrip(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result.export_visuals(
            export_dir=str(out_dir),
            file_name=file_stem,
            export_format=fmt,
        )
        return
    except TypeError:
        result.export_visuals(
            export_dir=str(out_dir),
            file_name=file_stem,
        )

    if fmt in ("png",):
        return

    png_path = out_dir / f"{file_stem}.png"
    if not png_path.exists():
        return

    img = cv2.imread(str(png_path))
    if img is None:
        return

    out_path = out_dir / f"{file_stem}.{fmt}"
    cv2.imwrite(str(out_path), img)
    png_path.unlink(missing_ok=True)


def main():
    args = build_args()

    source = Path(args.source)
    out_dir = Path(args.project)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(source, recursive=args.recursive)
    if not images:
        raise SystemExit(f"No supported images found in: {source}")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model_path,
        confidence_threshold=args.conf,
        device=args.device,
    )

    pbar_images = tqdm(
        total=len(images),
        desc="SAHI images",
        unit="img",
        position=1,
        leave=True,
        bar_format=BAR_FORMAT,
        colour=PURPLE,
        dynamic_ncols=True,
    )

    try:
        for img_path in images:
            pbar_crops = tqdm(
                total=1,
                desc=f"{img_path.name} crops",
                unit="crop",
                position=0,
                leave=False,
                bar_format=BAR_FORMAT,
                colour=PURPLE,
                dynamic_ncols=True,
            )

            result = get_sliced_prediction(
                image=str(img_path),
                detection_model=detection_model,
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.overlap_height_ratio,
                overlap_width_ratio=args.overlap_width_ratio,
            )

            file_stem = safe_unique_name(img_path)
            export_visual(result, out_dir, file_stem, args.format)

            pbar_crops.close()
            pbar_images.update(1)

    finally:
        pbar_images.close()

if __name__ == "__main__":
    main()
