"""Shared constants for the synthetic dataset generation pipeline."""

MAX_SIDE = 1024       # Max side after resize (preserves aspect ratio)
DIVISOR = 16          # FLUX requires dimensions as multiples of 16

MIN_OBJECTS = 2
MAX_OBJECTS = 3
MIN_DIST_PX = 120     # Minimum spacing between objects
EDGE_MARGIN = 60      # Safety margin from water boundaries

# Object sizes in pixels (for resized image with max side ~1024px).
# (min_w, max_w, min_h, max_h)
OBJECT_SIZES = {
    0: (50, 100, 25, 50),     # plastic bottle - elongated
    1: (50, 100, 25, 50),     # glass bottle - similar
    2: (40, 70,  35, 65),     # can - more square
    3: (80, 150, 60, 120),    # plastic bag - larger, spread out
    4: (50, 100, 40, 80),     # metal scrap - irregular
    5: (60, 110, 40, 80),     # plastic wrapper - rectangular
    6: (120, 200, 90, 160),   # trash pile - largest
    7: (40, 80,  30, 60),     # trash - generic small
}

# Crop-based inpainting settings.
# CROP_CONTEXT_FACTOR controls how much background context surrounds the object.
# Lower = object is a larger fraction of the crop → more detail, better generation.
# At 2.5×: a 60px object → 150px raw → MIN_CROP_SIZE=256 → object is ~23% of width.
# At 4.0×: same object → 240px raw → 320px min  → object is ~19% of width (too small).
CROP_CONTEXT_FACTOR = 2.5
MIN_CROP_SIZE = 256
MAX_CROP_SIZE = 640
