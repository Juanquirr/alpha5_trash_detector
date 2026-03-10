#!/usr/bin/env python3
import os
import json
import shutil
import argparse
import yaml
from pathlib import Path
from PIL import Image
from datetime import datetime


def yolo_to_coco_bbox(x_center, y_center, w, h, img_w, img_h):
    x_min = max(0.0, (x_center - w / 2) * img_w)
    y_min = max(0.0, (y_center - h / 2) * img_h)
    abs_w = min(w * img_w, img_w - x_min)
    abs_h = min(h * img_h, img_h - y_min)
    return [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)]


def convert_split(split_name, input_dir, output_dir, categories):
    images_src = Path(input_dir) / split_name / "images"
    labels_src = Path(input_dir) / split_name / "labels"

    if not images_src.exists():
        print(f"[SKIP] {split_name} not found")
        return

    images_dst = Path(output_dir) / "images" / split_name
    ann_dst    = Path(output_dir) / "annotations"
    images_dst.mkdir(parents=True, exist_ok=True)
    ann_dst.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": f"Dataset {split_name}", "date_created": datetime.now().strftime("%Y/%m/%d")},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    image_id = 1
    ann_id   = 1
    exts     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in sorted(f for f in images_src.iterdir() if f.suffix.lower() in exts):
        shutil.copy2(img_path, images_dst / img_path.name)

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        label_path = labels_src / (img_path.stem + ".txt")
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                bbox = yolo_to_coco_bbox(*map(float, parts[1:5]), img_w, img_h)
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 2),
                    "segmentation": [],
                    "iscrowd": 0,
                })
                ann_id += 1

        image_id += 1

    json_path = ann_dst / f"instances_{split_name}.json"
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"[{split_name}] images: {len(coco['images'])}, annotations: {len(coco['annotations'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve() if args.output else input_dir.parent / (input_dir.name + "_coco")

    with open(input_dir / "data.yaml") as f:
        data_yaml = yaml.safe_load(f)

    names = data_yaml.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    categories = [{"id": i + 1, "name": n, "supercategory": "trash"} for i, n in enumerate(names)]

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_dir / "data.yaml", output_dir / "data.yaml")

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Classes: {[c['name'] for c in categories]}")

    for split in ["train", "val", "test"]:
        convert_split(split, input_dir, output_dir, categories)

    print("Done.")


if __name__ == "__main__":
    main()

