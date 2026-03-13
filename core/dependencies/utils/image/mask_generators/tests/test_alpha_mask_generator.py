import numpy as np
import pytest

from PIL import Image, ImageDraw

from core.dependencies.utils.image.mask_generators.alpha_mask_generator import AlphaMaskGenerator


@pytest.fixture
def alpha_img():
    """Fixture: Creates a 10x10 alpha image with a filled square in the center."""
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(img)
    drawer.rectangle((4, 4, 6, 6), fill=(255, 0, 0, 255))
    return img

@pytest.fixture
def big_alpha_img():
    """Fixture: Creates a 14x14 alpha image with a larger filled square."""
    img = Image.new("RGBA", (14, 14), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(img)
    drawer.rectangle((2, 2, 8, 8), fill=(255, 0, 0, 255))
    return img

@pytest.fixture
def multiple_alpha_img():
    """Fixture: Creates a 10x10 alpha image with two separate filled squares."""
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(img)
    drawer.rectangle((2, 2, 4, 4), fill=(255, 0, 0, 255))
    drawer.rectangle((5, 5, 7, 7), fill=(255, 0, 0, 255))
    return img

def test_generate_inside_fill_mask(multiple_alpha_img):
    """Test: Check if the mask correctly fills the inside of the alpha area."""
    mask_generator = AlphaMaskGenerator(
        alpha_image=multiple_alpha_img,
        type=AlphaMaskGenerator.Type.inside
    )
    mask = mask_generator.generate()
    for x in range(10):
        for y in range(10):
            if multiple_alpha_img.getpixel((x, y))[3] != 0:  # Non-transparent pixels
                assert mask.getpixel((x, y)) == 255, f"Failed at position {(x, y)}"
            else:
                assert mask.getpixel((x, y)) == 0, f"Failed at position {(x, y)}"

def test_generate_outside_mask(multiple_alpha_img):
    """Test: Check if the mask correctly fills the outside of the alpha area."""
    mask_generator = AlphaMaskGenerator(
        alpha_image=multiple_alpha_img,
        type=AlphaMaskGenerator.Type.outside
    )
    mask = mask_generator.generate()
    for x in range(10):
        for y in range(10):
            if multiple_alpha_img.getpixel((x, y))[3] == 0:  # Transparent pixels
                assert mask.getpixel((x, y)) == 255, f"Failed at position {(x, y)}"
            else:
                assert mask.getpixel((x, y)) == 0, f"Failed at position {(x, y)}"

def test_generate_outside_mask_with_strength(multiple_alpha_img):
    mask_generator = AlphaMaskGenerator(
        alpha_image=multiple_alpha_img,
        type=AlphaMaskGenerator.Type.outside,
        strength=100
    )
    mask = mask_generator.generate()
    for x in range(10):
        for y in range(10):
            if multiple_alpha_img.getpixel((x, y))[3] == 0:  # Transparent pixels
                assert mask.getpixel((x, y)) == 100, f"Failed at position {(x, y)}"
            else:
                assert mask.getpixel((x, y)) == 0, f"Failed at position {(x, y)}"


def test_generate_border_inside_mask(big_alpha_img):
    """Test: Verify that the border inside mask marks only the inner edge of the alpha area."""
    mask_generator = AlphaMaskGenerator(
        alpha_image=big_alpha_img,
        type=AlphaMaskGenerator.Type.border_inside,
        border_width=2
    )
    mask = mask_generator.generate()
    mask_array = np.array(mask)

    # Verifica que haya valores > 0 solo en los bordes internos
    for x in range(14):
        for y in range(14):
            alpha_value = big_alpha_img.getpixel((x, y))[3]
            if alpha_value > 0:  # Dentro de la región alpha
                # Comprueba si está en el borde interno (dentro del ancho definido)
                if any(
                    big_alpha_img.getpixel((x + dx, y + dy))[3] == 0
                    for dx in range(-2, 3)
                    for dy in range(-2, 3)
                    if 0 <= x + dx < 14 and 0 <= y + dy < 14
                ):
                    assert mask_array[y, x] == 255, f"Expected border at ({x}, {y})"
                else:
                    assert mask_array[y, x] == 0, f"Unexpected border at ({x}, {y})"

def test_generate_border_inside_mask_with_strength(big_alpha_img):
    """Test: Verify that the border inside mask marks only the inner edge of the alpha area."""
    mask_generator = AlphaMaskGenerator(
        alpha_image=big_alpha_img,
        type=AlphaMaskGenerator.Type.border_inside,
        border_width=2,
        strength=126
    )
    mask = mask_generator.generate()
    mask_array = np.array(mask)

    # Verifica que haya valores > 0 solo en los bordes internos
    for x in range(14):
        for y in range(14):
            alpha_value = big_alpha_img.getpixel((x, y))[3]
            if alpha_value > 0:  # Dentro de la región alpha
                # Comprueba si está en el borde interno (dentro del ancho definido)
                if any(
                        big_alpha_img.getpixel((x + dx, y + dy))[3] == 0
                        for dx in range(-2, 3)
                        for dy in range(-2, 3)
                        if 0 <= x + dx < 14 and 0 <= y + dy < 14
                ):
                    assert mask_array[y, x] == 126, f"Expected border at ({x}, {y})"
                else:
                    assert mask_array[y, x] == 0, f"Unexpected border at ({x}, {y})"

def test_generate_border_outside_mask(big_alpha_img):
    """Test: Verify that the border outside mask marks only the outer edge of the alpha area."""
    mask_generator = AlphaMaskGenerator(
        alpha_image=big_alpha_img,
        type=AlphaMaskGenerator.Type.border_outside,
        border_width=1
    )
    mask = mask_generator.generate()
    mask_array = np.array(mask)

    # Verifica que haya valores > 0 solo en los bordes externos
    for x in range(14):
        for y in range(14):
            alpha_value = big_alpha_img.getpixel((x, y))[3]
            if alpha_value == 0:  # Fuera de la región alpha
                # Comprueba si está justo en el borde externo
                if any(
                    big_alpha_img.getpixel((x + dx, y + dy))[3] > 0
                    for dx in range(-1, 2)
                    for dy in range(-1, 2)
                    if 0 <= x + dx < 14 and 0 <= y + dy < 14
                ):
                    assert mask_array[y, x] > 0, f"Expected border at ({x}, {y})"
                else:
                    assert mask_array[y, x] == 0, f"Unexpected border at ({x}, {y})"

def test_generate_border_outside_mask_with_strenght(big_alpha_img):
    """Test: Verify that the border outside mask marks only the outer edge of the alpha area."""
    mask_generator = AlphaMaskGenerator(
        alpha_image=big_alpha_img,
        type=AlphaMaskGenerator.Type.border_outside,
        border_width=1,
        strength=100
    )
    mask = mask_generator.generate()
    mask_array = np.array(mask)

    # Verifica que haya valores > 0 solo en los bordes externos
    for x in range(14):
        for y in range(14):
            alpha_value = big_alpha_img.getpixel((x, y))[3]
            if alpha_value == 0:  # Fuera de la región alpha
                # Comprueba si está justo en el borde externo
                if any(
                        big_alpha_img.getpixel((x + dx, y + dy))[3] > 0
                        for dx in range(-1, 2)
                        for dy in range(-1, 2)
                        if 0 <= x + dx < 14 and 0 <= y + dy < 14
                ):
                    assert mask_array[y, x] == 100, f"Expected border at ({x}, {y})"
                else:
                    assert mask_array[y, x] == 0, f"Unexpected border at ({x}, {y})"
