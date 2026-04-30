import csv
import datetime
import hashlib
from pathlib import Path

# Prompt versions — full text stored here, only hash saved in CSV.
# To recover prompt text: PROMPT_REGISTRY[prompt_hash]
PROMPT_REGISTRY: dict[str, str] = {}

FIELDS = [
    "timestamp",
    "image",
    "model",
    "variant",
    "prompt_hash",   # short hash instead of full prompt text
    "response",
    "garbage_detected",
    "classes_detected",
    "inference_s",
    "vram_mb",
]


def _prompt_hash(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode()).hexdigest()[:8]
    PROMPT_REGISTRY[h] = prompt
    return h


def already_processed(csv_path: str) -> set[str]:
    """Return set of image filenames already present in csv_path."""
    path = Path(csv_path)
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # support both old 'prompt' column and new 'prompt_hash'
        return {row["image"] for row in reader if "image" in row}


def append_row(row: dict, csv_path: str = "results/detections.csv") -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    row["timestamp"]   = datetime.datetime.now().isoformat(timespec="seconds")
    row["prompt_hash"] = _prompt_hash(row.pop("prompt", ""))

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_prompt_registry(results_dir: str = "results") -> None:
    """Write prompts.txt so full prompt text is never lost."""
    path = Path(results_dir) / "prompts.txt"
    with open(path, "w", encoding="utf-8") as f:
        for h, text in PROMPT_REGISTRY.items():
            f.write(f"[{h}]\n{text}\n\n")
