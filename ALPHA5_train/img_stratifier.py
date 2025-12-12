import os
import random
import shutil
import argparse
from collections import defaultdict

random.seed(42)
IMG_EXTS = (".jpg", ".jpeg", ".png")


def count_yolo_instances(label_path: str) -> dict[int, int]:
    counts = defaultdict(int)
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            counts[class_id] += 1
    return dict(counts)


def load_samples(mixed_dir: str):
    samples = []
    for img_name in os.listdir(mixed_dir):
        if not img_name.lower().endswith(IMG_EXTS):
            continue

        stem, _ = os.path.splitext(img_name)
        label_name = stem + ".txt"
        label_path = os.path.join(mixed_dir, label_name)
        if not os.path.exists(label_path):
            continue

        counts = count_yolo_instances(label_path)
        if not counts:
            continue

        samples.append((img_name, label_name, counts))
    return samples


def stratified_instance_split(samples, split_ratios=(0.7, 0.2, 0.1)):
    total_counts = defaultdict(int)
    for _, _, counts in samples:
        for c, n in counts.items():
            total_counts[c] += n
    total_counts = dict(total_counts)

    target_train = {c: int(total_counts[c] * split_ratios[0]) for c in total_counts}
    target_val = {c: int(total_counts[c] * split_ratios[1]) for c in total_counts}
    target_test = {c: total_counts[c] - target_train[c] - target_val[c] for c in total_counts}

    used_train = defaultdict(int)
    used_val = defaultdict(int)
    used_test = defaultdict(int)

    indices = list(range(len(samples)))
    random.shuffle(indices)

    assigned = {"train": set(), "val": set(), "test": set()}

    def can_add(i, split):
        _, _, counts = samples[i]
        used = used_train if split == "train" else used_val if split == "val" else used_test
        target = target_train if split == "train" else target_val if split == "val" else target_test
        for c, n in counts.items():
            if used[c] + n > target.get(c, 0):
                return False
        return True

    def add(i, split):
        _, _, counts = samples[i]
        used = used_train if split == "train" else used_val if split == "val" else used_test
        for c, n in counts.items():
            used[c] += n
        assigned[split].add(i)

    for split in ("train", "val", "test"):
        for i in indices:
            if i in assigned["train"] or i in assigned["val"] or i in assigned["test"]:
                continue
            if can_add(i, split):
                add(i, split)

    for i in indices:
        if i in assigned["train"] or i in assigned["val"] or i in assigned["test"]:
            continue

        def fill_score(split):
            used = used_train if split == "train" else used_val if split == "val" else used_test
            target = target_train if split == "train" else target_val if split == "val" else target_test
            ratios = []
            for c in total_counts:
                t = target.get(c, 0)
                if t > 0:
                    ratios.append(min(1.0, used[c] / t))
            return sum(ratios) / max(1, len(ratios))

        scores = {s: fill_score(s) for s in ("train", "val", "test")}
        best_split = min(scores, key=scores.get)

        if can_add(i, best_split):
            add(i, best_split)
        else:
            add(i, "train")

    train_set = [samples[i] for i in assigned["train"]]
    val_set = [samples[i] for i in assigned["val"]]
    test_set = [samples[i] for i in assigned["test"]]

    used = {"train": dict(used_train), "val": dict(used_val), "test": dict(used_test)}
    targets = {"train": target_train, "val": target_val, "test": target_test}

    return train_set, val_set, test_set, total_counts, used, targets


def copy_split(mixed_dir: str, out_dir: str, split_name: str, dataset):
    img_out = os.path.join(out_dir, split_name, "images")
    lbl_out = os.path.join(out_dir, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img_name, lbl_name, _ in dataset:
        shutil.copy(os.path.join(mixed_dir, img_name), os.path.join(img_out, img_name))
        shutil.copy(os.path.join(mixed_dir, lbl_name), os.path.join(lbl_out, lbl_name))


def balance_by_instance_splits(mixed_dir: str, out_dir: str, split_ratios=(0.7, 0.2, 0.1)):
    samples = load_samples(mixed_dir)
    train_set, val_set, test_set, totals, used, targets = stratified_instance_split(samples, split_ratios)

    for split_name, ds in (("train", train_set), ("val", val_set), ("test", test_set)):
        copy_split(mixed_dir, out_dir, split_name, ds)

    print("Total instances per class:", totals)
    print("Targets:", targets)
    print("Used:", used)

    return used


def build_args():
    p = argparse.ArgumentParser(description="Instance-stratified dataset split for YOLO labels")
    p.add_argument("mixed_dir", type=str, help="Folder containing images and YOLO .txt labels")
    p.add_argument("--out_dir", type=str, default=None, help="Output folder (default: <mixed_dir>/balanced_by_instances)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.2, 0.1),
                   help="Split ratios: train val test (e.g. --ratios 0.7 0.2 0.1)")
    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    random.seed(args.seed)

    out_dir = args.out_dir or os.path.join(args.mixed_dir, "balanced_by_instances")
    stats = balance_by_instance_splits(args.mixed_dir, out_dir, split_ratios=tuple(args.ratios))
    print("Final instances per split:", stats)
