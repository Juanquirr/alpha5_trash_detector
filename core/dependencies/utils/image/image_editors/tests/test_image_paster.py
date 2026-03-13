from matplotlib import pyplot as plt
import pytest
from PIL import Image

from core.dependencies.utils.image.image_editors.image_paster import ImagePaster


@pytest.fixture
def sample_image():
    image = Image.new("RGBA", (10, 10), (0, 255, 255, 255))
    return image

@pytest.fixture
def sample_patch():
    patch = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
    return patch

def test_image_paster(sample_image, sample_patch):
    paster = ImagePaster(
        patch=sample_patch,
        center=(5, 5)
    )
    img = paster.edit(sample_image)
    assert img.getpixel((5, 5)) == (255, 0, 0, 255), "pixels inside center should be colored"
    assert img.getpixel((4, 5)) == (255, 0, 0, 255)
    assert img.getpixel((5, 4)) == (255, 0, 0, 255)
    assert img.getpixel((4, 4)) == (255, 0, 0, 255)
    assert img.getpixel((6, 6)) == (0, 255, 255, 255)
    assert img.getpixel((7, 7)) == (0, 255, 255, 255)


def test_image_paster_when_patch_outside_image_borders(sample_image, sample_patch):
    paster = ImagePaster(
        patch=sample_patch,
        center=(0, 0)
    )
    img = paster.edit(sample_image)
    assert img.getpixel((0, 0)) == (255, 0, 0, 255), "pixels inside center should be colored"
    assert img.getpixel((1, 1)) == (0, 255, 255, 255)

def test_paster_with_transparent_images(sample_image, sample_patch):
    patch = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    paster = ImagePaster(
        patch=patch,
        center=(0, 0)
    )
    img = paster.edit(sample_image)
    assert img.getpixel((0, 0)) == (0, 255, 255, 255), "pixels inside center should be colored"
    assert img.getpixel((1, 1)) == (0, 255, 255, 255)
