import pytest

from core.dependencies.utils.image.mask_generators.box_mask_generator import BoxMaskGenerator
from core.dependencies.utils.image.mask_generators.combined_mask_generator import MaskOperator


@pytest.fixture
def mask_one():
    """
    Create a 10x10 mask with a 2x2 box centered at (7, 7).
    """
    mask_generator = BoxMaskGenerator(
        output_size=(10, 10),
        mask_size=(2, 2),
        center=(7, 7)
    )
    return mask_generator

@pytest.fixture
def mask_two():
    """
    Create a 10x10 mask with a 2x2 box centered at (2, 2).
    """
    mask_generator = BoxMaskGenerator(
        output_size=(10, 10),
        mask_size=(2, 2),
        center=(2, 2)
    )
    return mask_generator

def test_combined_mask_generator(mask_one, mask_two):
    """
    Test combining two masks and validate pixel values:
    - Where mask_one is non-zero, combined mask should match mask_one.
    - Where mask_two is non-zero and mask_one is zero, combined mask should match mask_two.
    - Where both masks are zero, combined mask should remain zero.
    """
    img1 = mask_one.generate()
    img2 = mask_two.generate()
    mask_generator = MaskOperator(resolution=(10, 10))
    mask = mask_generator.combine(mask_one).combine(mask_two).generate()

    for x in range(0, 10):
        for y in range(0, 10):
            if img1.getpixel((x, y)) > 0:
                assert mask.getpixel((x, y)) == img1.getpixel((x, y)), "Mask should match img1 where img1 > 0"
            elif img2.getpixel((x, y)) > 0:
                assert mask.getpixel((x, y)) == img2.getpixel((x, y)), "Mask should match img2 where img2 > 0 and img1 == 0"
            else:
                assert mask.getpixel((x, y)) == 0, "Mask should be 0 where both img1 and img2 are 0"

def test_empty_combination():
    """
    Test generating a mask with no inputs.
    The output should be an empty (zero-filled) mask.
    """
    mask_generator = MaskOperator(resolution=(10, 10))
    mask = mask_generator.generate()

    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            assert mask.getpixel((x, y)) == 0, "Mask should be empty when no masks are combined"

def test_partial_overlap():
    """
    Test combining two masks with partial overlap.
    Verify that overlapping areas take the value of the first mask added.
    """
    mask_one = BoxMaskGenerator(output_size=(10, 10), mask_size=(4, 4), center=(5, 5))
    mask_two = BoxMaskGenerator(output_size=(10, 10), mask_size=(4, 4), center=(6, 6))

    img1 = mask_one.generate()
    img2 = mask_two.generate()

    mask_generator = MaskOperator(resolution=(10, 10))
    mask = mask_generator.combine(mask_one).combine(mask_two).generate()

    for x in range(0, 10):
        for y in range(0, 10):
            if img1.getpixel((x, y)) > 0:
                assert mask.getpixel((x, y)) == img1.getpixel((x, y)), "Mask should prioritize first mask in overlapping areas"
            elif img2.getpixel((x, y)) > 0:
                assert mask.getpixel((x, y)) == img2.getpixel((x, y)), "Mask should match img2 where img1 is zero"
            else:
                assert mask.getpixel((x, y)) == 0, "Mask should be 0 where both masks are 0"

def test_different_sizes():
    """
    Test combining masks of different sizes.
    An exception should be raised as sizes must match.
    """
    mask_one = BoxMaskGenerator(output_size=(10, 10), mask_size=(2, 2), center=(5, 5))
    mask_two = BoxMaskGenerator(output_size=(8, 8), mask_size=(2, 2), center=(4, 4))

    mask_generator = MaskOperator(resolution=(10, 10))
    with pytest.raises(ValueError):
        mask_generator.combine(mask_one).combine(mask_two).generate()

def test_no_overlap():
    """
    Test combining two masks with no overlapping areas.
    Verify that the combined mask contains values from both masks in their respective areas.
    """
    mask_one = BoxMaskGenerator(output_size=(10, 10), mask_size=(2, 2), center=(1, 1))
    mask_two = BoxMaskGenerator(output_size=(10, 10), mask_size=(2, 2), center=(8, 8))

    img1 = mask_one.generate()
    img2 = mask_two.generate()

    mask_generator = MaskOperator(resolution=(10, 10))
    mask = mask_generator.combine(mask_one).combine(mask_two).generate()

    for x in range(0, 10):
        for y in range(0, 10):
            if img1.getpixel((x, y)) > 0:
                assert mask.getpixel((x, y)) == img1.getpixel((x, y)), "Mask should match img1 where img1 > 0"
            elif img2.getpixel((x, y)) > 0:
                assert mask.getpixel((x, y)) == img2.getpixel((x, y)), "Mask should match img2 where img2 > 0"
            else:
                assert mask.getpixel((x, y)) == 0, "Mask should be 0 where both masks are 0"
