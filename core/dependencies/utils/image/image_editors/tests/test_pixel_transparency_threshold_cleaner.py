import pytest
from PIL import Image
import numpy as np

from matplotlib import pyplot as plt

from core.dependencies.utils.image.image_editors.pixel_transparency_threshold_cleaner import \
    PixelTransparencyThresholdCleanEditor


@pytest.fixture
def clean_transparent_pixels():
    img = Image.new("RGBA", (10, 10), (0, 255, 255, 255))
    img.putpixel((5, 5), (0, 0, 0, 165))
    img.putpixel((1, 1), (0, 0, 0, 100))
    img.putpixel((9, 9), (0, 0, 0, 230))
    return img

def test_clean_transparent_pixels(clean_transparent_pixels):
    editor = PixelTransparencyThresholdCleanEditor(
        threshold=240,
        clean_color=(0, 255, 255, 255)
    )

    result = editor.edit( clean_transparent_pixels )
    for x in range(0, result.size[0]):
        for y in range(0, result.size[1]):
            assert result.getpixel((x, y)) == (0, 255, 255, 255)

def test_clean_transparent_pixels_with_alpha_channel(clean_transparent_pixels):
    editor = PixelTransparencyThresholdCleanEditor(
        threshold=220,
        clean_color=(0, 255, 255, 255)
    )
    result = editor.edit( clean_transparent_pixels )
    for x in range(0, result.size[0]):
        for y in range(0, result.size[1]):
            if x == 9 and y == 9:
                assert result.getpixel((x, y)) == (0, 0, 0, 230)
            else:
                assert result.getpixel((x, y)) == (0, 255, 255, 255)

def test_clean_transparent_pixels_with_alpha_channel_and_background_color(clean_transparent_pixels):
    editor = PixelTransparencyThresholdCleanEditor(
        threshold=20,
        clean_color=(0, 0, 0, 255)
    )
    result = editor.edit( clean_transparent_pixels )
    assert result.getpixel((5, 5)) == (0, 0, 0, 165)
    result.putpixel((5, 5), (0, 255, 255, 255))
    assert result.getpixel((1, 1)) == (0, 0, 0, 100)
    result.putpixel((1, 1), (0, 255, 255, 255))
    assert result.getpixel((9, 9)) == (0, 0, 0, 230)
    result.putpixel((9, 9), (0, 255, 255, 255))
    for x in range(0, result.size[0]):
        for y in range(0, result.size[1]):
            assert result.getpixel((x, y)) == (0, 255, 255, 255)



