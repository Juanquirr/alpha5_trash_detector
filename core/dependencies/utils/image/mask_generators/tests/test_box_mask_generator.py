import numpy as np
import pytest

from core.dependencies.utils.image.mask_generators.box_mask_generator import BoxMaskGenerator


def test_box_mask_generator():
    """Test standard box mask generation with a centered mask."""
    mask_generator = BoxMaskGenerator(
        output_size=(10, 10),
        mask_size=(4, 4),
        center=(5, 5)
    )
    mask = mask_generator.generate()

    assert mask.size == (10, 10), "Mask size should be 10x10."
    assert mask.getpixel((5, 5)) == 255, "Center pixel (5, 5) should be 255."
    assert mask.getpixel((3, 3)) == 255, "Pixel (3, 3) should be within the mask and equal 255."
    assert mask.getpixel((2, 2)) == 0, "Pixel (2, 2) should be outside the mask and equal 0."
    assert mask.getpixel((7, 7)) == 255, "Pixel (7, 7) should be within the mask and equal 255."
    assert mask.getpixel((8, 8)) == 0, "Pixel (8, 8) should be outside the mask and equal 0."

def test_mask_size_is_greater_than_output_size():
    """Test when mask size exceeds the output image dimensions."""
    mask_generator = BoxMaskGenerator(
        output_size=(10, 10),
        mask_size=(12, 12),
        center=(5, 5)
    )
    mask = mask_generator.generate()

    assert mask.size == (10, 10), "Mask size should be clipped to the output size (10x10)."
    assert np.all(np.array(mask) == 255), "All pixels should be 255 as the mask covers the full output size."

def test_mask_just_one_pixel():
    """Test generating a mask with a single-pixel size."""
    mask_generator = BoxMaskGenerator(
        output_size=(10, 10),
        mask_size=(1, 1),
        center=(5, 5)
    )
    mask = mask_generator.generate()
    mask_array = np.array(mask)

    assert mask.size == (10, 10), "Mask size should be 10x10."
    assert mask.getpixel((5, 5)) == 255, "Single pixel at (5, 5) should be 255."
    assert mask_array[5, 5] == 255, "Mask array value at (5, 5) should be 255."

    mask_array[5, 5] = 0  # Temporarily zero out the center pixel
    assert np.all(mask_array == 0), "All pixels except (5, 5) should be 0."

def test_mask_at_top_left_corner():
    """Test mask placement when the center is at the top-left corner."""
    mask_generator = BoxMaskGenerator(
        output_size=(10, 10),
        mask_size=(4, 4),
        center=(0, 0)
    )
    mask = mask_generator.generate()
    mask_array = np.array(mask)

    assert mask.size == (10, 10), "Mask size should remain 10x10."
    assert mask_array[0:2, 0:2].sum() == 255 * 4, "Top-left pixels should be filled with 255."
    assert np.all(mask_array[4:, :] == 0), "Pixels outside the mask (bottom rows) should be 0."

def test_mask_at_bottom_right_corner():
    """Test mask placement when the center is at the bottom-right corner."""
    mask_generator = BoxMaskGenerator(
        output_size=(10, 10),
        mask_size=(4, 4),
        center=(9, 9)
    )
    mask = mask_generator.generate()
    mask_array = np.array(mask)

    assert mask.size == (10, 10), "Mask size should remain 10x10."
    assert mask_array[8:10, 8:10].sum() == 255 * 4, "Bottom-right pixels should be filled with 255."
    assert np.all(mask_array[:6, :] == 0), "Pixels outside the mask (top rows) should be 0."

def test_invalid_mask_size():
    """Test behavior when mask size is invalid (e.g., negative or zero)."""
    with pytest.raises(ValueError):
        mask_generator = BoxMaskGenerator(
            output_size=(10, 10),
            mask_size=(0, 0),
            center=(5, 5)
        )
        mask_generator.generate()
