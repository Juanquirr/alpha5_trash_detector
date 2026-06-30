"""
pope_build.py  —  Build POPE-style binary question files from YOLO annotations.

Reads YOLO .txt labels and writes three JSONL question files,
one per tier (random, popular, adversarial).

Balance: n_neg = n_pos per image (POPE standard ~50/50 yes/no ratio).
Negatives are ordered by tier strategy, then sliced to n_pos — so each tier
selects a DIFFERENT subset of negatives, making tier difficulty meaningful.
Clean images (0 GT classes) get CLEAN_IMAGE_NEGATIVES=3 negative questions.

Output line format:
    {"question_id": 1, "image": "foto1.jpg", "cls": "container",
     "text": "Is there a rigid non-metal container such as a bottle or jar in this image? Answer yes or no.",
     "label": "yes"}

Tier negative selection:
    random      — negatives shuffled randomly per image (seeded, reproducible)
    popular     — most-frequent classes in dataset selected first
    adversarial — classes that co-occur most with GT classes selected first (hardest)

Usage:
    python pope_build.py --dataset ../detection/datasets/alpha10
    python pope_build.py --images images/ --out pope_questions/
    python pope_build.py --images imgs/ --labels lbls/ --out pope_questions/
    python pope_build.py --seed 0
"""

import argparse
import datetime
import json
import random
from collections import defaultdict
from pathlib import Path

# Default 7-class taxonomy (used when no data.yaml is available)
DEFAULT_CLASSES = [
    "container",
    "plastic",
    "metal",
    "polystyrene",
    "plastic fragment",
    "trash pile",
    "trash",
]

# Per-class descriptive prompts — more informative than bare class name.
# Each prompt ends with "Answer yes or no." for consistent parsing.
CLASS_PROMPTS: dict[str, str] = {
    "container":
        "Is there a rigid non-metal container with an elongated or cylindrical shape,"
        " such as a plastic or glass bottle, jar, or rigid cup,"
        " in this image? Answer yes or no.",
    "plastic":
        "Is there flat, flexible, translucent plastic material such as a plastic bag,"
        " plastic film, or soft plastic wrapper floating in this image? Answer yes or no.",
    "metal":
        "Is there any item with a specular metallic reflection such as a metal can,"
        " aluminium foil, or metal scrap in this image? Answer yes or no.",
    "polystyrene":
        "Is there white opaque matte foam material such as EPS foam blocks,"
        " polystyrene cups, polystyrene plates, or white foam debris"
        " in this image? Answer yes or no.",
    "plastic fragment":
        "Is there a small compact rigid plastic piece such as a bottle cap, broken"
        " plastic fragment, plastic cutlery, or straw in this image? Answer yes or no.",
    "trash pile":
        "Is there a dense cluster or accumulation of multiple waste objects forming"
        " a heap of mixed garbage where individual items may be indistinguishable"
        " in this image? Answer yes or no.",
    "trash":
        "Is there a single unclassifiable waste item that cannot be identified as a"
        " container, plastic material, metal, polystyrene, plastic fragment, or trash"
        " pile in this image? Answer yes or no.",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SPLITS     = ("train", "val", "test")


def _read_data_yaml(dataset_root: Path) -> list[str] | None:
    """Read class names from data.yaml if present. Normalize underscores to spaces."""
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except ImportError:
        # Fallback: parse names line manually (covers [cls1, cls2, ...] format)
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("names:"):
                raw = line.split(":", 1)[1].strip()
                raw = raw.strip("[]")
                names = [n.strip().strip("'\"").replace("_", " ") for n in raw.split(",")]
                return [n for n in names if n]
        return None
    if "names" in data:
        names = data["names"]
        if isinstance(names, list):
            return [str(n).replace("_", " ") for n in names]
        if isinstance(names, dict):
            return [str(names[k]).replace("_", " ") for k in sorted(names)]
    return None


# ── Label loading ─────────────────────────────────────────────────────────────

def load_yolo_labels(
    images_dir: Path,
    labels_dir: Path | None = None,
    id_to_class: dict[int, str] | None = None,
) -> list[dict]:
    """
    Return list of {image: str, gt_classes: set[str]} from YOLO .txt files.

    If labels_dir is provided, reads .txt from there (standard YOLO split
    layout where images/ and labels/ are separate directories).
    Otherwise falls back to looking for .txt files next to images (flat layout).
    """
    if labels_dir is None:
        labels_dir = images_dir
    if id_to_class is None:
        id_to_class = {i: c for i, c in enumerate(DEFAULT_CLASSES)}

    records = []
    for txt in sorted(labels_dir.glob("*.txt")):
        # Find matching image file in images_dir
        image_name = txt.with_suffix(".jpg").name
        for ext in IMAGE_EXTS:
            candidate = images_dir / txt.with_suffix(ext).name
            if candidate.exists():
                image_name = candidate.name
                break

        content = txt.read_text().strip()
        gt_classes: set[str] = set()
        if content:
            for line in content.splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        cid = int(parts[0])
                        if cid in id_to_class:
                            gt_classes.add(id_to_class[cid])
                    except ValueError:
                        pass

        records.append({"image": image_name, "gt_classes": gt_classes})
    return records


def load_yolo_dataset(
    dataset_root: Path,
    id_to_class: dict[int, str] | None = None,
) -> tuple[list[dict], list[Path]]:
    """
    Load labels from a standard YOLO dataset directory (train/val/test splits).

    Expects structure:
        dataset_root/
            train/images/  train/labels/
            val/images/    val/labels/   (or valid/)
            test/images/   test/labels/  (optional)

    Returns (records, images_dirs) where images_dirs lists all image
    directories found (for metadata).
    """
    records: list[dict] = []
    images_dirs: list[Path] = []

    for split in SPLITS:
        split_images = dataset_root / split / "images"
        split_labels = dataset_root / split / "labels"
        if not split_images.exists() or not split_labels.exists():
            continue
        split_records = load_yolo_labels(split_images, split_labels, id_to_class)
        if split_records:
            records.extend(split_records)
            images_dirs.append(split_images)
            print(f"  [{split:5s}]  {len(split_records)} images")

    return records, images_dirs


# ── Question builders ─────────────────────────────────────────────────────────

CLEAN_IMAGE_NEGATIVES = 3   # negatives asked per clean image (0 GT classes)


def build_questions(records: list[dict], tier: str, seed: int = 42,
                    classes: list[str] | None = None) -> list[dict]:
    """
    Generate POPE binary questions for one tier.

    Per-image generation:
      - Labeled images: n_pos positives + n_neg = min(n_pos, available) negatives
      - Clean images (0 GT): CLEAN_IMAGE_NEGATIVES negative-only questions
        (pure hallucination test — model should always say "no")
      - Negatives ordered by tier strategy before slicing

    Post-hoc 50/50 balance (POPE standard):
      Clean images add excess negatives. After generating all questions,
      subsample "no" to match "yes" count for exact 50/50.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    def _get_prompt(cls_name: str) -> str:
        if cls_name in CLASS_PROMPTS:
            return CLASS_PROMPTS[cls_name]
        # Generic fallback for classes without a hand-written prompt
        return f"Is there a {cls_name} in this image? Answer yes or no."

    # Class frequency across dataset (for popular/adversarial tiers)
    class_freq: dict[str, int] = defaultdict(int)
    for r in records:
        for cls in r["gt_classes"]:
            class_freq[cls] += 1

    # Co-occurrence matrix: class_cooccur[a][b] = images where both a and b appear
    class_cooccur: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        gt = list(r["gt_classes"])
        for i, a in enumerate(gt):
            for b in gt[i + 1:]:
                class_cooccur[a][b] += 1
                class_cooccur[b][a] += 1

    rng = random.Random(seed)
    questions: list[dict] = []

    for r in records:
        gt_classes = r["gt_classes"]
        positives  = [c for c in classes if c in gt_classes]
        negatives  = [c for c in classes if c not in gt_classes]

        # Order negatives according to tier
        if tier == "random":
            rng.shuffle(negatives)
        elif tier == "popular":
            negatives.sort(key=lambda c: -class_freq.get(c, 0))
        elif tier == "adversarial":
            def _cooccur_score(c: str) -> int:
                return sum(class_cooccur[c].get(gt_cls, 0) for gt_cls in gt_classes)
            negatives.sort(key=_cooccur_score, reverse=True)

        n_neg = len(positives) if positives else CLEAN_IMAGE_NEGATIVES
        selected_negatives = negatives[:n_neg]

        for cls in positives + selected_negatives:
            label = "yes" if cls in gt_classes else "no"
            questions.append({
                "image": r["image"],
                "cls":   cls,
                "text":  _get_prompt(cls),
                "label": label,
            })

    # ── Post-hoc 50/50 balance (POPE standard) ──────────────────────────────
    yes_qs = [q for q in questions if q["label"] == "yes"]
    no_qs  = [q for q in questions if q["label"] == "no"]

    if len(no_qs) > len(yes_qs):
        rng2 = random.Random(seed + 1)
        rng2.shuffle(no_qs)
        no_qs = no_qs[:len(yes_qs)]
    elif len(yes_qs) > len(no_qs):
        rng2 = random.Random(seed + 1)
        rng2.shuffle(yes_qs)
        yes_qs = yes_qs[:len(no_qs)]

    balanced = yes_qs + no_qs
    random.Random(seed + 2).shuffle(balanced)

    # Assign sequential question_ids after balancing
    for qid, q in enumerate(balanced, 1):
        q["question_id"] = qid

    return balanced


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build POPE question JSONL files")
    parser.add_argument("--dataset", default=None,
                        help="YOLO dataset root (with train/images + train/labels). "
                             "Reads all splits automatically. Mutually exclusive with --images.")
    parser.add_argument("--images", default=None,
                        help="Flat directory with images + YOLO .txt labels side by side. "
                             "Ignored when --dataset is provided.")
    parser.add_argument("--labels", default=None,
                        help="Separate labels directory (used with --images when labels "
                             "are not next to images). Ignored when --dataset is provided.")
    parser.add_argument("--out",    default="pope_questions",
                        help="Output directory for JSONL files (created if missing)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for 'random' tier (default: 42)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load records from dataset or flat images dir ─────────────────────────
    classes = DEFAULT_CLASSES

    if args.dataset:
        dataset_root = Path(args.dataset)
        if not dataset_root.exists():
            print(f"ERROR: Dataset path not found: {dataset_root}")
            return
        # Read classes from data.yaml if available
        yaml_classes = _read_data_yaml(dataset_root)
        if yaml_classes:
            classes = yaml_classes
            print(f"Classes from data.yaml ({len(classes)}): {classes}")
        else:
            print(f"No data.yaml found, using default {len(DEFAULT_CLASSES)} classes")
        id_to_class = {i: c for i, c in enumerate(classes)}
        print(f"Loading YOLO dataset from {dataset_root} ...")
        records, images_dirs = load_yolo_dataset(dataset_root, id_to_class)
        images_dir_meta = [str(d.resolve()) for d in images_dirs]
    else:
        images_dir = Path(args.images or "images")
        labels_dir = Path(args.labels) if args.labels else None
        label_src  = labels_dir or images_dir
        print(f"Loading labels from {label_src} (images: {images_dir}) ...")
        records    = load_yolo_labels(images_dir, labels_dir)
        images_dir_meta = [str(images_dir.resolve())]

    if not records:
        print(f"ERROR: No .txt annotation files found.")
        return

    n_images  = len(records)
    n_labeled = sum(1 for r in records if r["gt_classes"])
    n_clean   = n_images - n_labeled
    print(f"  Total: {n_images} images  |  {n_labeled} with labels  |  {n_clean} clean\n")

    # Write metadata so pope_run.py / pope_finetune_eval.py can locate images
    meta = {
        "images_dirs": images_dir_meta,
        "classes":     classes,
        "built_at":    datetime.datetime.now().isoformat(timespec="seconds"),
        "n_images":    n_images,
        "seed":        args.seed,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  [metadata  ]  images = {images_dir_meta}")
    print(f"               saved  = {meta_path}\n")

    for tier in ("random", "popular", "adversarial"):
        questions = build_questions(records, tier=tier, seed=args.seed, classes=classes)
        out_path  = out_dir / f"pope_{tier}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")
        n_yes = sum(1 for q in questions if q["label"] == "yes")
        n_no  = len(questions) - n_yes
        print(
            f"  [{tier:<12s}]  {len(questions):5d} questions  "
            f"|  {n_yes} yes  |  {n_no} no  ->  {out_path}"
        )

    print(f"\nDone. Run pope_run.py --model <model> --tier all to start inference.")


if __name__ == "__main__":
    main()
