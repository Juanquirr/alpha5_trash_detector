from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class Alpha5Dataset(CocoDataset):
    """Dataset for Alpha5."""
    CLASSES = ('plastic_bottle', 'glass', 'can', 'plastic_bag', 'metal_scrap', 'plastic_wrapper',
         'trash_pile', 'trash')