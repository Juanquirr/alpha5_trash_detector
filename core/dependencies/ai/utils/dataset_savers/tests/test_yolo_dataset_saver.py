import os

from PIL import Image
import uuid

from core.dependencies.ai.utils.dataset_savers.yolo_dataset_saver import YoloDatasetSaver

import shutil

def test_yolo_generates_directories_correctly():
    dataset_saver = YoloDatasetSaver(boat_category=0)
    img1 = Image.new("RGB", (64, 64), (0, 0, 0, 0))
    bbox = (0.5, 0.5, 0.5, 0.5)
    dataset_saver.add_training(img1, bbox)
    dataset_saver.add_training(img1, bbox)
    id = str(uuid.uuid4())
    dataset_saver.save(id)
    assert id in os.listdir(), "Dataset saver has not created directory dataset"
    assert "images" in os.listdir(id), "Dataset must create images folder"
    assert "train" in os.listdir(id + "/images"), "Dataset must create train images folder"
    assert "val" in os.listdir(id + "/images"), "Dataset must create validation images folder"
    assert "labels" in os.listdir(id), "Dataset must create images folder"
    assert "train" in os.listdir(id + "/labels"), "Dataset must create train labels folder"
    assert "val" in os.listdir(id + "/labels"), "Dataset must create validation labels folder"
    assert len(os.listdir(id + "/images/train")) == 1, "Dataset is not creating images in train directory"
    assert len(os.listdir(id + "/labels/train")) == 1, "Dataset is not creating labels in train directory"
    shutil.rmtree(id)


