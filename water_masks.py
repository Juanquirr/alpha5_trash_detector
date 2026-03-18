import argparse
import glob
import os
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

METHODS = {
    "hsv":    "core.water_detector",
    "otsu":   "core.water_detector_otsu",
    "kmeans": "core.water_detector_kmeans",
    "flood":  "core.water_detector_flood",
}

parser = argparse.ArgumentParser(description="Generate water masks using different detection methods.")
parser.add_argument(
    "--method", choices=list(METHODS.keys()), default="hsv",
    help="Water detection method (default: hsv)",
)
parser.add_argument("--limit", type=int, default=10, help="Max images to process (default: 10)")
args = parser.parse_args()

import importlib
module = importlib.import_module(METHODS[args.method])
create_water_mask = module.create_water_mask

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"water_masks_{args.method}_{timestamp}"
os.makedirs(out_dir, exist_ok=True)

images = sorted(glob.glob("inputs/*.jpeg") + glob.glob("inputs/*.jpg"))[:args.limit]
print(f"Method: {args.method} | Output: {out_dir} | Images: {len(images)}")

with tqdm(total=len(images), desc="Water masks", unit="img", position=0, leave=True) as pbar:
    for p in images:
        img = np.array(Image.open(p).convert("RGB"))
        mask = create_water_mask(img)
        pct = mask.mean() / 255
        name = os.path.basename(p)
        Image.fromarray(mask).save(os.path.join(out_dir, f"{name}_water.png"))
        pbar.set_postfix({"last": name[:30], "water": f"{pct:.0%}"})
        pbar.update(1)
