from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from typing import Tuple, List, Dict, Any


def plot_images(images, titles, main_title="Images", save_as="result.png"):
    num_images = len(images)
    plt.figure(figsize=(5 * num_images, 5))
    plt.suptitle(main_title, fontsize=16)

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, num_images, i + 1)  # 1 row, num_images columns, ith subplot
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.show()
    plt.savefig(save_as)

def draw_square_inside_image(
        image: Image.Image,
        square_size: Tuple[int, int],
        center_point: Tuple[int, int],
        border_width: int = 3,
        center_radius: int = 0,
        color="red",
) -> Image.Image:
    image_with_square = image.copy()
    draw = ImageDraw.Draw(image_with_square)
    half_width, half_height = square_size[0] // 2, square_size[1] // 2
    top_left = (center_point[0] - half_width, center_point[1] - half_height)
    bottom_right = (center_point[0] + half_width, center_point[1] + half_height)
    for i in range(border_width):
        draw.rectangle(
            [
                (top_left[0] - i, top_left[1] - i),
                (bottom_right[0] + i, bottom_right[1] + i)
            ],
            outline=color
        )

    if center_radius > 0:
        draw.ellipse(
            [
                (center_point[0] - center_radius, center_point[1] - center_radius),
                (center_point[0] + center_radius, center_point[1] + center_radius)
            ],
            fill=color
        )

    return image_with_square

def polygon_to_bbox(polygon: List[List[float]]) -> Dict[str, float]:
    """
    Converts a polygon (list of normalized [x, y] points) to a normalized bounding box.

    Args:
        polygon: List of [x, y] normalized coordinates.

    Returns:
        A dictionary with keys: 'left', 'top', 'width', 'height'
        All values are in normalized [0, 1] range.
    """
    if not polygon:
        raise ValueError("Polygon must contain at least one point.")

    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    return {
        "point": [min_x, min_y],
        "width": max_x - min_x,
        "height": max_y - min_y
    }

import random
from typing import List
from PIL import Image

def choose_random_elements(patches: List[Any], n: int) -> List[Any]:
    """
    Randomly selects `n` patches from the list.

    Args:
        patches (List[PIL.Image]): List of patch images.
        n (int): Number of patches to select.

    Returns:
        List[PIL.Image]: Randomly selected patches.
    """
    if n >= len(patches):
        return patches  # return all if asking for more than available
    return random.sample(patches, n)

from PIL import Image, ImageDraw

def paint_masks_in_image(image: Image.Image, masks: list, color=(255, 0, 0), alpha=128):
    """
    Paints semi-transparent colored boxes on the original image using the given masks.

    Parameters:
        image (PIL.Image): Original image (RGB or RGBA).
        masks (list of PIL.Image): List of grayscale masks (white = region to paint).
        color (tuple): RGB color for the boxes.
        alpha (int): Alpha transparency (0 = transparent, 255 = opaque).

    Returns:
        PIL.Image: Image with masks painted on top.
    """
    # Ensure image is RGBA
    base = image.convert("RGBA")

    for mask in masks:
        # Create a color image with transparency, shaped by the mask
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha_mask = mask.point(lambda p: alpha if p > 0 else 0)
        overlay.putalpha(alpha_mask)

        # Composite the overlay onto the base image
        base = Image.alpha_composite(base, overlay)

    return base


from PIL import Image
import numpy as np
from typing import List, Tuple

def divide_images_with_masks_and_bboxes(img: Image.Image, n_divisions: int = 1, padding: int = 10) -> Tuple[List[Image.Image], np.ndarray]:
    """
    Divide a PIL image into a grid and return binary masks and bounding boxes.

    Parameters:
        img (PIL.Image): Input image.
        n_divisions (int): Number of divisions per axis (total masks = n_divisions^2).
        padding (int): Padding inside each region to shrink the white area.

    Returns:
        Tuple:
            - List[PIL.Image.Image]: List of mask images (white = region, black = elsewhere).
            - np.ndarray: Array of bounding boxes in (x1, y1, x2, y2) format.
    """
    width, height = img.size
    masks = []
    bounding_boxes = []

    cell_w = width // n_divisions
    cell_h = height // n_divisions

    for i in range(n_divisions):
        for j in range(n_divisions):
            x1 = j * cell_w
            y1 = i * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            # Apply padding safely
            pad_x1 = min(width, max(0, x1 + padding))
            pad_y1 = min(height, max(0, y1 + padding))
            pad_x2 = min(width, max(0, x2 - padding))
            pad_y2 = min(height, max(0, y2 - padding))

            # Create binary mask
            mask_array = np.zeros((height, width), dtype=np.uint8)
            mask_array[pad_y1:pad_y2, pad_x1:pad_x2] = 255

            # Convert to PIL Image
            mask_img = Image.fromarray(mask_array, mode="L")
            masks.append(mask_img)

            # Save bounding box
            bounding_boxes.append([mask_img, (pad_x1, pad_y1, pad_x2, pad_y2)])

    return bounding_boxes


from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw

def filter_masks_by_polygon(
    patches: List[Image.Image],
    polygon: List[List[float]],
    threshold: float = 0.3
) -> List[Image.Image]:
    """
    Filters full-image-sized binary patch masks by overlap with a normalized polygon.

    Args:
        patches (List[PIL.Image]): List of binary masks (white patch area).
        polygon (List[List[float]]): List of normalized [x, y] points.
        threshold (float): Minimum overlap ratio to keep the patch.

    Returns:
        List[PIL.Image]: Filtered masks with enough polygon overlap.
    """
    if not patches:
        return []

    width, height = patches[0][0].size

    # Convert normalized polygon to absolute coordinates
    poly_pixels = [(int(x * width), int(y * height)) for x, y in polygon]

    # Create polygon mask over full image size
    poly_mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(poly_mask).polygon(poly_pixels, fill=1)
    poly_mask_np = np.array(poly_mask)

    filtered = []

    for patch in patches:
        patch_np = np.array(patch[0]) > 0  # True where patch is white
        intersection = np.logical_and(patch_np, poly_mask_np).sum()
        patch_area = patch_np.sum()

        if patch_area == 0:
            continue

        overlap_ratio = intersection / patch_area

        if overlap_ratio >= threshold:
            filtered.append(patch)

    return filtered


def plug_annotation(image: Image.Image, bounding_box: dict) -> dict:
    """
    Converts bounding box and image info into a normalized annotation payload.

    Args:
        image (str): Reference to the image or image data string.
        bounding_box (dict): Bounding box with keys 'x_0', 'y_0', 'x_1', 'y_1'.
        info (dict): Contains original image size as 'size': {'width': ..., 'height': ...}.

    Returns:
        dict: Normalized annotation in the expected structure.
    """
    width = image.size[0]
    height = image.size[1]

    x0, y0, x1, y1 = bounding_box['x_0'], bounding_box['y_0'], bounding_box['x_1'], bounding_box['y_1']

    annotation = {
        "width": abs(x1 - x0) / width,
        "height": abs(y1 - y0) / height,
        "point": [x0 / width, y0 / height]
    }

    return annotation