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
    0: (35,  65,  18,  35),   # container       - elongated/cylindrical (bottles, jars)
    1: (50,  90,  35,  65),   # plastic          - flat, spread out (bags, film)
    2: (30,  60,  25,  50),   # metal            - variable (cans, foil, scrap)
    3: (50,  85,  40,  70),   # polystyrene      - chunky, medium-large (foam blocks)
    4: (15,  35,  12,  28),   # plastic_fragment - small, compact (caps, fragments)
    5: (75, 120,  60,  95),   # trash_pile       - largest (dense cluster)
    6: (30,  52,  22,  40),   # trash            - generic single item
}

# Crop-based inpainting settings.
# CROP_CONTEXT_FACTOR controls how much background context surrounds the object.
# Lower = object is a larger fraction of the crop → more detail, better generation.
# At 2.5×: a 60px object → 150px raw → MIN_CROP_SIZE=256 → object is ~23% of width.
# At 4.0×: same object → 240px raw → 320px min  → object is ~19% of width (too small).
CROP_CONTEXT_FACTOR = 2.5
MIN_CROP_SIZE = 256
MAX_CROP_SIZE = 640
