import os
import random
import shutil
import argparse
from collections import defaultdict

random.seed(42)
EXT_IMGS = (".jpg", ".jpeg", ".png")


def count_instances_yolo(annotation_path):
    counts = defaultdict(int)
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"Invalid class ID on line {line_num} of '{annotation_path}': "
                    f"expected an integer but got '{parts[0]}'"
                )
            counts[class_id] += 1
    return counts


def is_split_folder(folder):
    """True if folder has at least one train/val/test subdir with an images/ inside."""
    for split in ("train", "val", "test"):
        if os.path.isdir(os.path.join(folder, split, "images")):
            return True
    return False


def load_dataset_from_split(split_folder):
    """
    Accepts an already-split YOLO dataset (train/, val/, test/ each with images/ and labels/).
    Pools annotated images for re-stratification. Hard negatives (empty labels) are collected
    separately per split so they can be preserved in their original split on output.

    Returns:
      image_details, class_to_images, total_counts, sources, hard_negatives_by_split
      hard_negatives_by_split: {split_name: [(img, ann, img_src, ann_src), ...]}
    """
    if not os.path.isdir(split_folder):
        raise FileNotFoundError(f"Input folder not found: '{split_folder}'")

    image_details = []
    class_to_images = defaultdict(list)
    total_counts = defaultdict(int)
    skipped = []
    sources = {}
    hard_negatives_by_split = {"train": [], "val": [], "test": []}

    found_splits = []
    for split in ("train", "val", "test"):
        img_dir = os.path.join(split_folder, split, "images")
        lbl_dir = os.path.join(split_folder, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        found_splits.append(split)

        for img in os.listdir(img_dir):
            if not img.lower().endswith(EXT_IMGS):
                continue
            base, _ = os.path.splitext(img)
            ann = base + ".txt"
            img_path = os.path.join(img_dir, img)
            ann_path = os.path.join(lbl_dir, ann)

            if not os.path.exists(ann_path):
                skipped.append((img, f"no annotation in {lbl_dir}"))
                continue

            counts = count_instances_yolo(ann_path)
            if not counts:
                hard_negatives_by_split[split].append((img, ann, img_path, ann_path))
                continue

            if img in sources:
                skipped.append((img, "duplicate filename already loaded from another split"))
                continue

            idx = len(image_details)
            image_details.append((img, ann, counts))
            sources[img] = (img_path, ann_path)
            for c, count in counts.items():
                class_to_images[c].append(idx)
                total_counts[c] += count

    if not found_splits:
        raise FileNotFoundError(
            f"No train/val/test subdirs with images/ found in '{split_folder}'"
        )

    print(f"Loaded splits: {found_splits}")

    total_hn = sum(len(v) for v in hard_negatives_by_split.values())
    if total_hn:
        print(f"Hard negatives found (will be preserved in their original split): "
              + ", ".join(f"{s}={len(hard_negatives_by_split[s])}" for s in found_splits))

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} image(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not image_details:
        raise ValueError(f"No valid annotated image-annotation pairs found in '{split_folder}'")

    return image_details, class_to_images, total_counts, sources, hard_negatives_by_split


def load_dataset(mixed_folder):
    """
    Returns:
      image_details, class_to_images, total_counts, hard_negatives
      hard_negatives: [(img, ann, img_src, ann_src)] — images with empty label files
    """
    if not os.path.isdir(mixed_folder):
        raise FileNotFoundError(f"Input folder not found: '{mixed_folder}'")

    images = [f for f in os.listdir(mixed_folder) if f.lower().endswith(EXT_IMGS)]
    if not images:
        raise ValueError(f"No images found in '{mixed_folder}' (expected {EXT_IMGS})")

    image_details = []
    class_to_images = defaultdict(list)
    total_counts = defaultdict(int)
    skipped = []
    hard_negatives = []

    for idx, img in enumerate(images):
        base, _ = os.path.splitext(img)
        ann = base + ".txt"
        ann_path = os.path.join(mixed_folder, ann)

        if not os.path.exists(ann_path):
            skipped.append((img, "no annotation file"))
            continue

        counts = count_instances_yolo(ann_path)
        if not counts:
            hard_negatives.append((
                img, ann,
                os.path.join(mixed_folder, img),
                ann_path,
            ))
            continue

        image_details.append((img, ann, counts))

        for c, count in counts.items():
            class_to_images[c].append(idx)
            total_counts[c] += count

    if hard_negatives:
        print(f"Hard negatives found: {len(hard_negatives)} (will be distributed proportionally)")

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} image(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not image_details:
        raise ValueError(
            f"No valid annotated image-annotation pairs found in '{mixed_folder}'. "
            f"Checked {len(images)} image(s), all were skipped."
        )

    return image_details, class_to_images, total_counts, hard_negatives


def stratified_instance_split(image_details, class_to_images, total_counts,
                              split_ratios=(0.7, 0.2, 0.1)):
    """
    Tries to achieve, for each class c:
      train_instances[c] ≈ 0.7 * total_counts[c]
      val_instances[c]   ≈ 0.2 * total_counts[c]
      test_instances[c]  ≈ 0.1 * total_counts[c]
    using greedy image assignment.
    """
    target_train = {c: int(total_counts[c] * split_ratios[0]) for c in total_counts}
    target_val   = {c: int(total_counts[c] * split_ratios[1]) for c in total_counts}
    target_test  = {c: total_counts[c] - target_train[c] - target_val[c] for c in total_counts}

    used_train = defaultdict(int)
    used_val   = defaultdict(int)
    used_test  = defaultdict(int)

    num_images = len(image_details)
    indices = list(range(num_images))
    random.shuffle(indices)

    split_assign = {"train": set(), "val": set(), "test": set()}

    def is_assigned(idx):
        return (idx in split_assign["train"]
                or idx in split_assign["val"]
                or idx in split_assign["test"])

    def can_add(idx, split):
        _, _, counts = image_details[idx]
        if split == "train":
            return all(used_train[c] + n <= target_train[c] for c, n in counts.items())
        elif split == "val":
            return all(used_val[c] + n <= target_val[c] for c, n in counts.items())
        else:
            return all(used_test[c] + n <= target_test[c] for c, n in counts.items())

    def add(idx, split):
        _, _, counts = image_details[idx]
        if split == "train":
            for c, n in counts.items():
                used_train[c] += n
            split_assign["train"].add(idx)
        elif split == "val":
            for c, n in counts.items():
                used_val[c] += n
            split_assign["val"].add(idx)
        else:
            for c, n in counts.items():
                used_test[c] += n
            split_assign["test"].add(idx)

    # First pass: fill train, then val, then test
    for split in ("train", "val", "test"):
        for idx in indices:
            if is_assigned(idx):
                continue
            if can_add(idx, split):
                add(idx, split)

    # Second pass: assign any remaining images to the split furthest from its target
    targets = {"train": target_train, "val": target_val, "test": target_test}
    used    = {"train": used_train,   "val": used_val,   "test": used_test}

    for idx in indices:
        if is_assigned(idx):
            continue

        scores = {}
        for split in ("train", "val", "test"):
            ratios = [
                min(1.0, used[split][c] / targets[split][c])
                for c in total_counts
                if targets[split][c] > 0
            ]
            scores[split] = sum(ratios) / max(1, len(ratios))

        best_split = min(scores, key=scores.get)
        if can_add(idx, best_split):
            add(idx, best_split)
        else:
            add(idx, "train")

    train_set = [image_details[i] for i in split_assign["train"]]
    val_set   = [image_details[i] for i in split_assign["val"]]
    test_set  = [image_details[i] for i in split_assign["test"]]

    return train_set, val_set, test_set, (used_train, used_val, used_test), (target_train, target_val, target_test)


def copy_split(mixed_folder, output_folder, split_name, dataset, sources=None):
    img_out = os.path.join(output_folder, split_name, "images")
    lab_out = os.path.join(output_folder, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)

    for img, ann, _ in dataset:
        if sources and img in sources:
            img_src, ann_src = sources[img]
        else:
            img_src = os.path.join(mixed_folder, img)
            ann_src = os.path.join(mixed_folder, ann)
        shutil.copy(img_src, os.path.join(img_out, img))
        shutil.copy(ann_src, os.path.join(lab_out, ann))


def copy_hard_negatives_split(output_folder, hard_negatives_by_split):
    """Copy hard negatives into their original split, preserving the split assignment."""
    for split_name, negatives in hard_negatives_by_split.items():
        if not negatives:
            continue
        img_out = os.path.join(output_folder, split_name, "images")
        lab_out = os.path.join(output_folder, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lab_out, exist_ok=True)
        for img, ann, img_src, ann_src in negatives:
            shutil.copy(img_src, os.path.join(img_out, img))
            shutil.copy(ann_src, os.path.join(lab_out, ann))
        print(f"  Hard negatives → {split_name}: {len(negatives)}")


def distribute_hard_negatives_flat(output_folder, hard_negatives, split_ratios):
    """Distribute flat hard negatives proportionally across output splits."""
    if not hard_negatives:
        return
    n = len(hard_negatives)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])
    shuffled = list(hard_negatives)
    random.shuffle(shuffled)
    distribution = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }
    for split_name, negatives in distribution.items():
        if not negatives:
            continue
        img_out = os.path.join(output_folder, split_name, "images")
        lab_out = os.path.join(output_folder, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lab_out, exist_ok=True)
        for img, ann, img_src, ann_src in negatives:
            shutil.copy(img_src, os.path.join(img_out, img))
            shutil.copy(ann_src, os.path.join(lab_out, ann))
        print(f"  Hard negatives → {split_name}: {len(negatives)}")


def balance_by_instance_splits(mixed_folder, output_folder, split_ratios=(0.7, 0.2, 0.1), sources=None):
    image_details, class_to_images, total_counts, hard_negatives = load_dataset(mixed_folder)
    print("Total instances per class:", dict(total_counts))

    train_set, val_set, test_set, used_sets, target_sets = stratified_instance_split(
        image_details, class_to_images, total_counts, split_ratios
    )

    for name, ds in [("train", train_set), ("val", val_set), ("test", test_set)]:
        copy_split(mixed_folder, output_folder, name, ds, sources)

    distribute_hard_negatives_flat(output_folder, hard_negatives, split_ratios)

    used_train, used_val, used_test = used_sets
    target_train, target_val, target_test = target_sets

    print("Target train:", target_train, "| Used:", dict(used_train))
    print("Target val:  ", target_val,   "| Used:", dict(used_val))
    print("Target test: ", target_test,  "| Used:", dict(used_test))

    return {
        "train": dict(used_train),
        "val":   dict(used_val),
        "test":  dict(used_test),
    }


def build_args():
    p = argparse.ArgumentParser(
        description="Stratified train/val/test split for a YOLO dataset, balancing instances per class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_folder",
                   help="Flat folder (images + .txt side-by-side) OR already-split YOLO folder "
                        "(with train/val/test subdirs). Auto-detected unless --from-split is set.")
    p.add_argument("--output", "-o", default=None,
                   help="Output folder. Defaults to <input_folder>/balanced_by_instances.")
    p.add_argument("--train", type=float, default=0.7,
                   help="Fraction of instances to assign to train.")
    p.add_argument("--val",   type=float, default=0.2,
                   help="Fraction of instances to assign to val.")
    p.add_argument("--test",  type=float, default=0.1,
                   help="Fraction of instances to assign to test.")
    p.add_argument("--from-split", action="store_true",
                   help="Force treating input as an already-split YOLO dataset. "
                        "Auto-detected by default.")
    return p.parse_args()


def main():
    args = build_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(
            f"Split ratios must sum to 1.0, but got {args.train} + {args.val} + {args.test} = {total:.4f}"
        )

    input_folder  = args.input_folder
    output_folder = args.output or os.path.join(input_folder, "balanced_by_instances")
    split_ratios  = (args.train, args.val, args.test)

    use_split_loader = args.from_split or is_split_folder(input_folder)

    if use_split_loader:
        print(f"Detected split dataset structure in '{input_folder}'. Pooling all splits...")
        image_details, class_to_images, total_counts, sources, hard_negatives_by_split = \
            load_dataset_from_split(input_folder)
        print("Total instances per class:", dict(total_counts))

        train_set, val_set, test_set, used_sets, target_sets = stratified_instance_split(
            image_details, class_to_images, total_counts, split_ratios
        )
        for name, ds in [("train", train_set), ("val", val_set), ("test", test_set)]:
            copy_split(input_folder, output_folder, name, ds, sources)

        copy_hard_negatives_split(output_folder, hard_negatives_by_split)

        used_train, used_val, used_test = used_sets
        target_train, target_val, target_test = target_sets
    else:
        print(f"Flat folder detected. Loading images from '{input_folder}'...")
        stats = balance_by_instance_splits(input_folder, output_folder, split_ratios)
        print("Final instances per class and split:", stats)
        return

    print("Target train:", target_train, "| Used:", dict(used_train))
    print("Target val:  ", target_val,   "| Used:", dict(used_val))
    print("Target test: ", target_test,  "| Used:", dict(used_test))
    print("Final instances per class and split:", {
        "train": dict(used_train),
        "val":   dict(used_val),
        "test":  dict(used_test),
    })


if __name__ == "__main__":
    main()
