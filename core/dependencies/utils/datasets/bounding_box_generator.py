from PIL import Image, ImageDraw

from core.pipelines.pipeline import BoundingBox


class BoundingBoxGenerator:
    def generate(self, alpha_img) -> BoundingBox:
        alpha_data = alpha_img.getchannel("A")
        bbox = alpha_data.getbbox()
        if bbox:
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
            return BoundingBox(x=x, y=y, w=w, h=h)
        raise ValueError("No bounding box found")

    def paint(self, img, bbox: BoundingBox, color="red"):
        if bbox:
            draw = ImageDraw.Draw(img)
            draw.rectangle([bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h], outline=color, width=2)
        return img