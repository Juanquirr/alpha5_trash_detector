"""Prompt and class name loading from CSV configuration."""

import csv
import random

LIGHTING_MAPPING = {
    "morning": ", soft morning natural light, diffused ambient lighting, realistic look",
    "midday":  ", harsh midday sunlight, bright overhead lighting, high contrast shadows",
    "night":   ", dim night lighting, low ambient light, dark coastal atmosphere, realistic noise",
}

DEFAULT_LIGHTING = "midday"


def resolve_lighting(prompt: str, lighting: str | None = None) -> str:
    """Replace [LIGHTING] placeholder with the appropriate lighting string."""
    if "[LIGHTING]" not in prompt:
        return prompt
    key = lighting or DEFAULT_LIGHTING
    return prompt.replace("[LIGHTING]", LIGHTING_MAPPING.get(key, LIGHTING_MAPPING[DEFAULT_LIGHTING]))


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
