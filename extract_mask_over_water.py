import argparse
import glob
import os
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

METHODS = {
    "hsv":   "core.water_detector",
    "otsu":  "core.water_detector_otsu",
    "kmeans": "core.water_detector_kmeans",
    "flood": "core.water_detector_flood",
}

parser = argparse.ArgumentParser(description="Generate water masks using different detection methods.")
parser.add_argument(
    "--method", choices=list(METHODS.keys()), default="hsv",
    help="Water detection method (default: hsv)",
)
parser.add_argument("--limit", type=int, default=10, help="Max images to process (default: 10)")
args = parser.parse_args()

# Dynamic import of the selected method
import importlib
module = importlib.import_module(METHODS[args.method])
create_water_mask = module.create_water_mask

# Output folder: method + timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"water_masks_{args.method}_{timestamp}"
os.makedirs(out_dir, exist_ok=True)

images = sorted(glob.glob("inputs/*.jpeg") + glob.glob("inputs/*.jpg"))[:args.limit]
print(f"Method: {args.method} | Output: {out_dir} | Images: {len(images)}")

for p in images:
    img = np.array(Image.open(p).convert("RGB"))
    mask = create_water_mask(img)
    pct = mask.mean() / 255
    name = os.path.basename(p)
    print(f"{name[:50]:50s} water={pct:.0%}")
    Image.fromarray(mask).save(os.path.join(out_dir, f"{name}_water.png"))