import csv
import datetime
from pathlib import Path


def already_processed(csv_path: str) -> set[str]:
    """Return set of image filenames already present in csv_path."""
    path = Path(csv_path)
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["image"] for row in reader if "image" in row}

FIELDS = [
    "timestamp",
    "image",
    "model",
    "variant",
    "prompt",
    "response",
    "garbage_detected",
    "classes_detected",
    "inference_s",
    "vram_mb",
]


def append_row(row: dict, csv_path: str = "results/detections.csv") -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    row["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
