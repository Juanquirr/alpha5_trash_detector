"""
Water detection using SAM 3 (Segment Anything Model 3) with text prompts.

Strategy:
  Use SAM 3's open-vocabulary text prompting to directly segment water
  regions. Unlike heuristic-based detectors (HSV, Otsu, etc.), this
  approach leverages deep learning to understand what "water" actually
  looks like, regardless of color or lighting conditions.

  Text prompts: "water", "sea", "ocean" are used to capture different
  ways water may appear. All resulting masks are merged.

Requirements:
  pip install transformers torch
  Model: facebook/sam3 (downloaded automatically on first use)
  GPU: Required (runs on CUDA)
"""

import numpy as np
import torch
from PIL import Image

from core.water_detector import morphological_cleanup, remove_small_regions

# Text prompts to detect water regions — SAM3 handles open vocabulary
_WATER_PROMPTS = ["water", "sea", "ocean"]

# Singleton model cache (avoid reloading on every call)
_model = None
_processor = None


def _load_model(device: str = "cuda"):
    """Lazy-load SAM3 model and processor (singleton)."""
    global _model, _processor
    if _model is None:
        from transformers import AutoModel, AutoProcessor

        print("  Loading SAM3 model (first time may download ~3.5 GB)...")
        _processor = AutoProcessor.from_pretrained("facebook/sam3")
        _model = AutoModel.from_pretrained(
            "facebook/sam3",
            torch_dtype=torch.bfloat16,
        ).to(device)
        print("  SAM3 model loaded.")
    return _model, _processor


def create_water_mask(
    image_np: np.ndarray,
    min_region_ratio: float = 0.003,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cuda",
) -> np.ndarray:
    """
    Create a binary mask identifying water regions using SAM3 text prompts.

    Pipeline:
    1. Run SAM3 with text prompts ("water", "sea", "ocean").
    2. Merge all detected instance masks into a single water mask.
    3. Sky exclusion (top 10%).
    4. Morphological cleanup + small region removal.

    Args:
        image_np: RGB image as numpy array, shape (H, W, 3).
        min_region_ratio: Minimum connected region area as fraction of image.
        threshold: Confidence threshold for instance detection.
        mask_threshold: Threshold for binarizing predicted masks.
        device: Torch device ("cuda" or "cpu").

    Returns:
        Binary mask (H, W), uint8: 255 = water, 0 = non-water.
    """
    img_h, img_w = image_np.shape[:2]
    model, processor = _load_model(device)

    image_pil = Image.fromarray(image_np)
    water = np.zeros((img_h, img_w), dtype=np.uint8)

    # ── 1. Run SAM3 with each water-related text prompt ────────────
    for prompt in _WATER_PROMPTS:
        inputs = processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        # ── 2. Merge all detected masks ────────────────────────────
        for mask_tensor in results["masks"]:
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            water = np.maximum(water, mask_np)

    # ── 3. Sky exclusion ───────────────────────────────────────────
    water[: int(img_h * 0.10), :] = 0

    # ── 4. Morphological cleanup + small region removal ────────────
    water = morphological_cleanup(water)
    water = remove_small_regions(water, min_region_ratio)

    return water
