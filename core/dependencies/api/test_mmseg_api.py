from PIL import Image

from matplotlib import pyplot as plt

from core.dependencies.api.mmseg_api import MMSegAPI


def test_mmseg_api_image_is_segmented():
    img = Image.open("../../../data_assets/boats/with_bg/image_1")
    segmentation = MMSegAPI(url="http://100.103.218.9:4553/v1").segment_image(img)
    assert segmentation is not None

def test_mmseg_api_image_is_cached():
    img = Image.new("RGB", (256, 256))
    segmentation = MMSegAPI(url="http://100.103.218.9:4553/v1").segment_image(img)
    assert segmentation is not None
