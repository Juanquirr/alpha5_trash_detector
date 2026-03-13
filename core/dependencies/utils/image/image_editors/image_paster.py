from PIL import Image
from typing import Tuple

from pydantic import ConfigDict

from core.dependencies.utils.image.image_editors.image_editor import ImageEditor


class ImagePaster(ImageEditor):
    patch: Image.Image
    center: Tuple[int, int]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def edit(self, original_image: Image.Image) -> Image.Image:
        result_image = original_image.copy()
        cx, cy = self.center

        # Calculate cropping and pasting boundaries
        left = cx - self.patch.width // 2
        top = cy - self.patch.height // 2
        right = left + self.patch.width
        bottom = top + self.patch.height

        # Ensure paste_box is within bounds
        paste_box = (
            max(0, left),
            max(0, top),
            min(right, original_image.width),
            min(bottom, original_image.height)
        )

        # Crop the patch to match the region being pasted
        crop_box = (
            paste_box[0] - left,
            paste_box[1] - top,
            paste_box[2] - left,
            paste_box[3] - top
        )

        # Crop patch and preserve transparency
        cropped_patch = self.patch.crop(crop_box)

        # Paste cropped patch onto the result image using alpha channel as a mask
        result_image.paste(cropped_patch, paste_box, cropped_patch)
        return result_image
