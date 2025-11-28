import os
import random
import shutil
from collections import defaultdict

random.seed(42)
EXT_IMGS = (".jpg", ".jpeg", ".png")


def count_instances_yolo(annotation_path):
    counts = defaultdict(int)
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            class_id = int(line.split()[0])
            counts[class_id] += 1
    return counts


def load_dataset(mixed_folder):
    """
    Devuelve:
      - image_details: [(img_name, ann_name, counts_por_clase), ...]
      - class_to_images: {class_id: [indices de image_details que contienen esa clase]}
      - total_counts: instancias totales por clase
    """
    images = [f for f in os.listdir(mixed_folder)
              if f.lower().endswith(EXT_IMGS)]

    image_details = []
    class_to_images = defaultdict(list)
    total_counts = defaultdict(int)

    for idx, img in enumerate(images):
        base, _ = os.path.splitext(img)
        ann = base + ".txt"
        ann_path = os.path.join(mixed_folder, ann)
        if not os.path.exists(ann_path):
            continue

        counts = count_instances_yolo(ann_path)
        if not counts:
            continue

        image_details.append((img, ann, counts))

        for c, n in counts.items():
            class_to_images[c].append(idx)
            total_counts[c] += n

    return image_details, class_to_images, total_counts


def stratified_instance_split(image_details, class_to_images, total_counts,
                              split_ratios=(0.7, 0.2, 0.1)):
    """
    Intenta conseguir, para cada clase c:
      instancias_train[c] ≈ 0.7 * total_counts[c]
      instancias_val[c]   ≈ 0.2 * total_counts[c]
      instancias_test[c]  ≈ 0.1 * total_counts[c]
    usando asignación greedy de imágenes. 
    """
    # objetivos de instancias por clase y split
    target_train = {c: int(total_counts[c] * split_ratios[0]) for c in total_counts}
    target_val   = {c: int(total_counts[c] * split_ratios[1]) for c in total_counts}
    target_test  = {c: total_counts[c] - target_train[c] - target_val[c] for c in total_counts}

    # contadores actuales
    used_train = defaultdict(int)
    used_val   = defaultdict(int)
    used_test  = defaultdict(int)

    n = len(image_details)
    indices = list(range(n))
    random.shuffle(indices)

    split_assign = {"train": set(), "val": set(), "test": set()}

    def can_add(idx, split):
        img, ann, counts = image_details[idx]
        if split == "train":
            for c, n in counts.items():
                if used_train[c] + n > target_train[c]:
                    return False
        elif split == "val":
            for c, n in counts.items():
                if used_val[c] + n > target_val[c]:
                    return False
        else:  # test
            for c, n in counts.items():
                if used_test[c] + n > target_test[c]:
                    return False
        return True

    def add(idx, split):
        img, ann, counts = image_details[idx]
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

    # estrategia simple: primera pasada para train, luego val, luego test
    for split, target, used in [("train", target_train, used_train),
                                ("val", target_val, used_val),
                                ("test", target_test, used_test)]:
        for idx in indices:
            if idx in split_assign["train"] or idx in split_assign["val"] or idx in split_assign["test"]:
                continue
            if can_add(idx, split):
                add(idx, split)

    # cualquier imagen no asignada aún va al split con mayor "hueco" global
    for idx in indices:
        if idx in split_assign["train"] or idx in split_assign["val"] or idx in split_assign["test"]:
            continue
        # elegir split con menos fracción de cumplimiento de objetivos
        scores = {}
        for split, target, used in [("train", target_train, used_train),
                                    ("val", target_val, used_val),
                                    ("test", target_test, used_test)]:
            # suma de proporciones usadas/objetivo
            s = 0.0
            k = 0
            for c in total_counts:
                if target[c] > 0:
                    s += min(1.0, used[c] / target[c])
                    k += 1
            scores[split] = s / max(1, k)
        best_split = min(scores, key=scores.get)
        if can_add(idx, best_split):
            add(idx, best_split)
        else:
            # si no cabe en ninguno según objetivos, lo mandamos a train por defecto
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
        shutil.copy(os.path.join(mixed_folder, img),
                    os.path.join(img_out, img))
        shutil.copy(os.path.join(mixed_folder, ann),
                    os.path.join(lab_out, ann))


def balance_by_instance_splits(mixed_folder, output_folder,
                               split_ratios=(0.7, 0.2, 0.1)):
    image_details, class_to_images, total_counts = load_dataset(mixed_folder)
    print("Instancias totales por clase:", dict(total_counts))

    train_set, val_set, test_set, used_sets, target_sets = stratified_instance_split(
        image_details, class_to_images, total_counts, split_ratios
    )

    # copiar archivos
    for name, ds in [("train", train_set), ("val", val_set), ("test", test_set)]:
        copy_split(mixed_folder, output_folder, name, ds)

    # mostrar stats
    used_train, used_val, used_test = used_sets
    target_train, target_val, target_test = target_sets
    print("Objetivos train:", target_train)
    print("Usado   train:", dict(used_train))
    print("Objetivos val:", target_val)
    print("Usado   val:", dict(used_val))
    print("Objetivos test:", target_test)
    print("Usado   test:", dict(used_test))

    return {
        "train": dict(used_train),
        "val": dict(used_val),
        "test": dict(used_test),
    }


if __name__ == "__main__":
    mixed_folder = r"imagesv3.3"
    output_folder = os.path.join(mixed_folder, "balanced_by_instances")

    stats = balance_by_instance_splits(mixed_folder, output_folder)
    print("Instancias finales por clase y split:", stats)
