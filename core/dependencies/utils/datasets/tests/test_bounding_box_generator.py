import pytest
from PIL import Image, ImageDraw

from core.dependencies.utils.datasets.bounding_box_generator import BoundingBoxGenerator
from core.pipelines.pipeline import BoundingBox


@pytest.fixture
def generator():
    return BoundingBoxGenerator()

def test_generate_with_alpha_content(generator):
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 30, 70, 80], fill=(255, 255, 255, 255))
    bbox = generator.generate(img)
    assert bbox == BoundingBox(x=20, y=30, w=51, h=51)


def test_generate_with_no_alpha_content(generator):
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    bbox = generator.generate(img)
    assert bbox is None

def test_paint_with_bbox(generator):
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 30, 70, 80], fill=(255, 255, 255, 255))
    bbox = generator.generate(img)
    result_img = generator.paint(img.copy(), bbox)
    assert result_img is not None

def test_paint_with_no_bbox(generator):
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    bbox = generator.generate(img)
    result_img = generator.paint(img.copy(), bbox)
    assert result_img is not None
