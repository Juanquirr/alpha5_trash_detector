from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class Alpha5Dataset(CocoDataset):
    CLASSES = ('container', 'plastic', 'metal', 'polystyrene',
               'plastic_fragment', 'trash_pile', 'trash')