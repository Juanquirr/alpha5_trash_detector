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


def load_dataset(mixed_folder):
    """
    Returns:
      - image_details: [(img_name, ann_name, counts_per_class), ...]
      - class_to_images: {class_id: [indices into image_details that contain that class]}
      - total_counts: total instances per class
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

    for idx, img in enumerate(images):
        base, _ = os.path.splitext(img)
        ann = base + ".txt"
        ann_path = os.path.join(mixed_folder, ann)

        if not os.path.exists(ann_path):
            skipped.append((img, "no annotation file"))
            continue

        counts = count_instances_yolo(ann_path)
        if not counts:
            skipped.append((img, "annotation file is empty"))
            continue

        image_details.append((img, ann, counts))

        for c, count in counts.items():
            class_to_images[c].append(idx)
            total_counts[c] += count

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} image(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not image_details:
        raise ValueError(
            f"No valid image-annotation pairs found in '{mixed_folder}'. "
            f"Checked {len(images)} image(s), all were skipped."
        )

    return image_details, class_to_images, total_counts


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

        # Pick the split with the lowest average fulfillment ratio
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
            # Overflow: assign to train as a fallback
            add(idx, "train")

    train_set = [image_details[i] for i in split_assign["train"]]
    val_set   = [image_details[i] for i in split_assign["val"]]
    test_set  = [image_details[i] for i in split_assign["test"]]

    return train_set, val_set, test_set, (used_train, used_val, used_test), (target_train, target_val, target_test)


def copy_split(mixed_folder, output_folder, split_name, dataset):
    img_out = os.path.join(output_folder, split_name, "images")
    lab_out = os.path.join(output_folder, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)

    for img, ann, _ in dataset:
        shutil.copy(os.path.join(mixed_folder, img), os.path.join(img_out, img))
        shutil.copy(os.path.join(mixed_folder, ann), os.path.join(lab_out, ann))


def balance_by_instance_splits(mixed_folder, output_folder, split_ratios=(0.7, 0.2, 0.1)):
    image_details, class_to_images, total_counts = load_dataset(mixed_folder)
    print("Total instances per class:", dict(total_counts))

    train_set, val_set, test_set, used_sets, target_sets = stratified_instance_split(
        image_details, class_to_images, total_counts, split_ratios
    )

    for name, ds in [("train", train_set), ("val", val_set), ("test", test_set)]:
        copy_split(mixed_folder, output_folder, name, ds)

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
        description="Stratified train/val/test split for a YOLO dataset, balancing instances per class."
    )
    p.add_argument("mixed_folder",
                   help="Folder containing images and their .txt YOLO annotation files side-by-side.")
    p.add_argument("--output", "-o", default=None,
                   help="Output folder. Defaults to <mixed_folder>/balanced_by_instances.")
    p.add_argument("--train", type=float, default=0.7,
                   help="Fraction of instances to assign to train (default: 0.7).")
    p.add_argument("--val",   type=float, default=0.2,
                   help="Fraction of instances to assign to val (default: 0.2).")
    p.add_argument("--test",  type=float, default=0.1,
                   help="Fraction of instances to assign to test (default: 0.1).")
    return p.parse_args()


def main():
    args = build_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(
            f"Split ratios must sum to 1.0, but got {args.train} + {args.val} + {args.test} = {total:.4f}"
        )

    mixed_folder  = args.mixed_folder
    output_folder = args.output or os.path.join(mixed_folder, "balanced_by_instances")

    stats = balance_by_instance_splits(
        mixed_folder, output_folder,
        split_ratios=(args.train, args.val, args.test)
    )
    print("Final instances per class and split:", stats)


if __name__ == "__main__":
    main()
