import pytest
from PIL import Image

from matplotlib import pyplot as plt

from core.dependencies.utils.image.image_editors.image_cropper import ImageCropper


@pytest.fixture
def sample_image():
    """Creates a sample image of size 10x10 with a white background and a black center pixel."""
    image = Image.new("RGBA", (10, 10), (0, 255, 255, 255))
    image.putpixel((5, 5), (0, 0, 0, 255))  # Mark center pixel black
    return image

def test_center_crop_within_bounds(sample_image):
    cropper = ImageCropper(center=(5, 5), resolution=(6, 6))
    cropped = cropper.edit(sample_image)
    assert cropped.size == (6, 6)
    assert cropped.getpixel((3, 3)) == (0, 0, 0, 255), "Center pixel should remain black."

def test_partial_crop_at_edge(sample_image):
    cropper = ImageCropper(center=(0, 0), resolution=(12, 12))
    cropped = cropper.edit(sample_image)
    assert cropped.size == (12, 12)
    assert cropped.getpixel((0, 0)) == (0, 0, 0, 0), "Top-left corner should be white."
    assert cropped.getpixel((11, 11)) == (0, 0, 0, 255), "Out-of-bounds area should be transparent."

def test_exact_image_crop(sample_image):
    cropper = ImageCropper(center=(5, 5), resolution=(10, 10))
    cropped = cropper.edit(sample_image)
    assert cropped.size == (10, 10)
    assert cropped.getpixel((5, 5)) == (0, 0, 0, 255), "Center pixel should remain black."

def test_crop_smaller_than_image(sample_image):
    cropper = ImageCropper(center=(5, 5), resolution=(3, 3))
    cropped = cropper.edit(sample_image)
    assert cropped.size == (3, 3)
    assert cropped.getpixel((1, 1)) == (0, 0, 0, 255), "Center pixel should be correctly cropped."

def test_crop_larger_than_image(sample_image):
    cropper = ImageCropper(center=(5, 5), resolution=(20, 20))
    cropped = cropper.edit(sample_image)
    assert cropped.size == (20, 20)
    assert cropped.getpixel((10, 10)) == (0, 0, 0, 255), "Out-of-bounds area should be transparent."

def test_corner_crop(sample_image):
    cropper = ImageCropper(center=(0, 0), resolution=(4, 4))
    cropped = cropper.edit(sample_image)
    assert cropped.size == (4, 4)
    assert cropped.getpixel((0, 0)) == (0, 0, 0, 0), "Top-left corner must be transparent."
    assert cropped.getpixel((3, 3)) == (0, 255, 255, 255), "Bounds inside image should be blue"
