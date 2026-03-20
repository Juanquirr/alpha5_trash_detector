"""Prompt and class name loading from CSV configuration."""

import csv


def load_prompts(csv_path: str) -> dict:
    """Load prompts grouped by class_id from a CSV file."""
    prompts_by_class = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            prompt = row["prompt"].strip().strip('"')
            prompts_by_class.setdefault(cid, []).append(prompt)
    return prompts_by_class


def load_class_names(csv_path: str) -> dict:
    """Load class name mapping {class_id: name} from a CSV file."""
    class_names = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            if cid not in class_names:
                class_names[cid] = row["class_name"].strip()
    return class_names
