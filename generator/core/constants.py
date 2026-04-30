"""Shared constants for the synthetic dataset generation pipeline."""

MAX_SIDE = 1024       # Max side after resize (preserves aspect ratio)
DIVISOR = 16          # FLUX requires dimensions as multiples of 16

MIN_OBJECTS = 2
MAX_OBJECTS = 3
MIN_DIST_PX = 60      # Minimum spacing between objects
EDGE_MARGIN = 30      # Safety margin from water boundaries

# Object sizes in pixels (for resized image with max side ~1024px).
# Sized for elevated harbour camera perspective: realistic floating trash
# occupies 3–7% of image width (~30–70px on a 1024px canvas).
# Depth scaling in pipeline.py (×0.5 to ×1.0) is applied on top of these.
# (min_w, max_w, min_h, max_h)
OBJECT_SIZES = {
    0: (35,  65,  18,  35),   # plastic bottle - elongated
    1: (35,  60,  18,  32),   # glass bottle - similar
    2: (35,  55,  30,  50),   # can - more square
    3: (50,  90,  35,  65),   # plastic bag - larger, spread out
    4: (35,  65,  28,  50),   # metal scrap - irregular
    5: (40,  70,  28,  50),   # plastic wrapper - rectangular
    6: (75, 120,  60,  95),   # trash pile - largest
    7: (30,  52,  22,  40),   # trash - generic small
}

# Crop-based inpainting settings.
# CROP_CONTEXT_FACTOR controls how much background context surrounds the object.
# Lower = object is a larger fraction of the crop → more detail, better generation.
# At 2.5×: a 60px object → 150px raw → MIN_CROP_SIZE=256 → object is ~23% of width.
# At 4.0×: same object → 240px raw → 320px min  → object is ~19% of width (too small).
CROP_CONTEXT_FACTOR = 2.5
MIN_CROP_SIZE = 256
MAX_CROP_SIZE = 640
