"""SAM3 singleton loader + per-class inference."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_model = None
_processor = None


def load_sam3(device: str, model_path: str = "facebook/sam3") -> tuple:
    global _model, _processor
    if _model is None:
        from transformers import Sam3Model, Sam3Processor

        print(f"[SAM3] Loading from '{model_path}' (first run may download ~3.5 GB)...")
        _processor = Sam3Processor.from_pretrained(model_path)
        _model = Sam3Model.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        _model.eval()
        print("[SAM3] Ready.\n")
    return _model, _processor


def run_prompts_for_class(
    image_pil: Image.Image,
    class_id: int,
    prompts: list[str],
    device: str,
    det_threshold: float,
    mask_threshold: float,
    model_path: str = "facebook/sam3",
    max_prompts: int = 0,
) -> list[tuple[np.ndarray, float, str, int]]:
    """
    Run prompts for one class. Returns list of (mask, score, prompt, class_id).
    max_prompts=0 means use all prompts.
    """
    model, processor = load_sam3(device, model_path)
    active = prompts[:max_prompts] if max_prompts > 0 else prompts
    instances: list[tuple[np.ndarray, float, str, int]] = []

    for prompt in active:
        inputs = processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=det_threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks  = results.get("masks",  [])
        scores = results.get("scores", [None] * len(masks))

        for mask_t, score_v in zip(masks, scores):
            mask_np = mask_t.cpu().numpy().astype(bool)
            score   = float(score_v) if score_v is not None else 1.0
            instances.append((mask_np, score, prompt, class_id))

    return instances
