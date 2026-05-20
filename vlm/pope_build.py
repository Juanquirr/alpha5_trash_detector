"""
pope_build.py  —  Build POPE-style binary question files from YOLO annotations.

Reads YOLO .txt labels from images/ and writes three JSONL question files,
one per tier (random, popular, adversarial).

Balance: n_neg = n_pos per image (POPE standard ~50/50 yes/no ratio).
Negatives are ordered by tier strategy, then sliced to n_pos — so each tier
selects a DIFFERENT subset of negatives, making tier difficulty meaningful.
Clean images (0 GT classes) get CLEAN_IMAGE_NEGATIVES=3 negative questions.

Output line format:
    {"question_id": 1, "image": "foto1.jpg", "cls": "plastic bottle",
     "text": "Is there a plastic bottle in this image? Answer with yes or no.",
     "label": "yes"}

Tier negative selection:
    random      — negatives shuffled randomly per image (seeded, reproducible)
    popular     — most-frequent classes in dataset selected first
    adversarial — classes that co-occur most with GT classes selected first (hardest)

Usage:
    python pope_build.py
    python pope_build.py --images images/ --out pope_questions/
    python pope_build.py --seed 0
"""

import argparse
import datetime
import json
import random
from collections import defaultdict
from pathlib import Path

CLASSES = [
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

YOLO_ID_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ── Label loading ─────────────────────────────────────────────────────────────

def load_yolo_labels(images_dir: Path) -> list[dict]:
    """Return list of {image: str, gt_classes: set[str]} from YOLO .txt files."""
    records = []
    for txt in sorted(images_dir.glob("*.txt")):
        # Find matching image file
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
                        if cid in YOLO_ID_TO_CLASS:
                            gt_classes.add(YOLO_ID_TO_CLASS[cid])
                    except ValueError:
                        pass

        records.append({"image": image_name, "gt_classes": gt_classes})
    return records


# ── Question builders ─────────────────────────────────────────────────────────

CLEAN_IMAGE_NEGATIVES = 3   # negatives asked per clean image (0 GT classes)


def build_questions(records: list[dict], tier: str, seed: int = 42) -> list[dict]:
    """
    Generate POPE binary questions for one tier.

    Balance: n_neg = n_pos per image (POPE standard 50/50 yes/no).
    Clean images (0 GT classes): CLEAN_IMAGE_NEGATIVES negatives, 0 positives.
    Negatives are ordered by tier strategy before slicing, so each tier
    selects a different subset → tier difficulty is meaningful.
    """
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
    qid = 1

    for r in records:
        gt_classes = r["gt_classes"]
        positives  = [c for c in CLASSES if c in gt_classes]
        negatives  = [c for c in CLASSES if c not in gt_classes]

        # Order negatives according to tier
        if tier == "random":
            rng.shuffle(negatives)
        elif tier == "popular":
            # Most-frequent class in dataset first → model most tempted to say YES
            negatives.sort(key=lambda c: -class_freq.get(c, 0))
        elif tier == "adversarial":
            # Classes that co-occur most with this image's GT classes first
            def _cooccur_score(c: str) -> int:
                return sum(class_cooccur[c].get(gt_cls, 0) for gt_cls in gt_classes)
            negatives.sort(key=_cooccur_score, reverse=True)

        # Select balanced subset: n_neg == n_pos.
        # Clean images use a fixed quota so they still test hallucination.
        n_neg = len(positives) if positives else CLEAN_IMAGE_NEGATIVES
        selected_negatives = negatives[:n_neg]

        for cls in positives + selected_negatives:
            label = "yes" if cls in gt_classes else "no"
            questions.append({
                "question_id": qid,
                "image":       r["image"],
                "cls":         cls,
                "text":        CLASS_PROMPTS[cls],
                "label":       label,
            })
            qid += 1

    return questions


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build POPE question JSONL files")
    parser.add_argument("--images", default="images",
                        help="Directory with images + YOLO .txt labels")
    parser.add_argument("--out",    default="pope_questions",
                        help="Output directory for JSONL files (created if missing)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for 'random' tier (default: 42)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels from {images_dir} ...")
    records = load_yolo_labels(images_dir)
    if not records:
        print(f"ERROR: No .txt annotation files found in {images_dir}")
        return

    n_images  = len(records)
    n_labeled = sum(1 for r in records if r["gt_classes"])
    n_clean   = n_images - n_labeled
    print(f"  {n_images} images  |  {n_labeled} with labels  |  {n_clean} clean\n")

    # Write metadata so pope_run.py can locate images without --images flag
    meta = {
        "images_dir": str(images_dir.resolve()),
        "built_at":   datetime.datetime.now().isoformat(timespec="seconds"),
        "n_images":   n_images,
        "seed":       args.seed,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  [metadata  ]  images_dir = {meta['images_dir']}  →  {meta_path}\n")

    for tier in ("random", "popular", "adversarial"):
        questions = build_questions(records, tier=tier, seed=args.seed)
        out_path  = out_dir / f"pope_{tier}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")
        n_yes = sum(1 for q in questions if q["label"] == "yes")
        n_no  = len(questions) - n_yes
        print(
            f"  [{tier:<12s}]  {len(questions):5d} questions  "
            f"|  {n_yes} yes  |  {n_no} no  →  {out_path}"
        )

    print(f"\nDone. Run pope_run.py --model <model> --tier all to start inference.")


if __name__ == "__main__":
    main()
