import cv2, numpy as np
from PIL import Image
from core.water_detector import create_water_mask
import glob

for p in sorted(glob.glob('inputs/*.jpeg') + glob.glob('inputs/*.jpg'))[:10]:
    img = np.array(Image.open(p).convert('RGB'))
    mask = create_water_mask(img)
    pct = mask.mean()/255
    print(f'{p.split("/")[-1][:50]:50s} water={pct:.0%}')
    Image.fromarray(mask).save(f'water_masks_2/{p.split("/")[-1]}_water.png')