import csv
import datetime
from pathlib import Path

FIELDS = [
    "timestamp",
    "image",
    "model",
    "variant",
    "prompt",
    "response",
    "garbage_detected",
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
