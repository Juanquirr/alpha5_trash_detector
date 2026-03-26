"""Shared constants for the synthetic dataset generation pipeline."""

MAX_SIDE = 1024       # Max side after resize (preserves aspect ratio)
DIVISOR = 16          # FLUX requires dimensions as multiples of 16

MIN_OBJECTS = 2
MAX_OBJECTS = 3
MIN_DIST_PX = 120     # Minimum spacing between objects
EDGE_MARGIN = 60      # Safety margin from water boundaries

# Object sizes in pixels (for resized image with max side ~1024px).
# Sized for full-image inpainting: FLUX needs the masked region to be
# large enough to generate recognisable detail on a 1024px canvas.
# Depth scaling in pipeline.py (×0.6 to ×1.4) is applied on top of these.
# (min_w, max_w, min_h, max_h)
OBJECT_SIZES = {
    0: (90,  160,  45,  90),   # plastic bottle - elongated
    1: (90,  150,  45,  80),   # glass bottle - similar
    2: (90,  140,  80, 130),   # can - more square
    3: (140, 240, 100, 180),   # plastic bag - larger, spread out
    4: (90,  160,  70, 130),   # metal scrap - irregular
    5: (100, 180,  70, 130),   # plastic wrapper - rectangular
    6: (190, 300, 150, 240),   # trash pile - largest
    7: (80,  130,  60, 100),   # trash - generic small
}

# Crop-based inpainting settings.
# CROP_CONTEXT_FACTOR controls how much background context surrounds the object.
# Lower = object is a larger fraction of the crop → more detail, better generation.
# At 2.5×: a 60px object → 150px raw → MIN_CROP_SIZE=256 → object is ~23% of width.
# At 4.0×: same object → 240px raw → 320px min  → object is ~19% of width (too small).
CROP_CONTEXT_FACTOR = 2.5
MIN_CROP_SIZE = 256
MAX_CROP_SIZE = 640
