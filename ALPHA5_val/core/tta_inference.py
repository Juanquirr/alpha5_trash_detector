"""
Test-Time Augmentation (TTA) method with several flip transforms and NMS fusion.
"""

import time

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

from alpha5_base_inference import InferenceMethod, InferenceResult
from wbf_utils import greedy_nms_classwise


class TTAInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "TTA",
            "Test-Time Augmentation with flip transforms",
        )
        self.default_params = {
            "conf": 0.25,
            "iou": 0.5,
            "tta_iou": 0.5,
            "imgsz": 640,
        }

    def apply_tta(self, image):
        h, w = image.shape[:2]
        augs = [
            (image.copy(), {"type": "original", "params": None}),
            (cv2.flip(image, 1), {"type": "flip_h", "params": {"width": w}}),
            (cv2.flip(image, 0), {"type": "flip_v", "params": {"height": h}}),
            (
                cv2.flip(image, -1),
                {"type": "flip_hv", "params": {"width": w, "height": h}},
            ),
        ]
        return augs

    def reverse_transform(self, boxes, transform_info):
        if len(boxes) == 0:
            return boxes
        boxes = np.array(boxes)
        ttype = transform_info["type"]
        params = transform_info["params"]

        if ttype == "original":
            return boxes
        elif ttype == "flip_h":
            w = params["width"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        elif ttype == "flip_v":
            h = params["height"]
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        elif ttype == "flip_hv":
            w, h = params["width"], params["height"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        return boxes

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)
        iou = params.get("iou", 0.5)
        imgsz = params.get("imgsz", 640)
        tta_iou = params.get("tta_iou", 0.5)

        augmentations = self.apply_tta(image)
        all_boxes, all_scores, all_classes = [], [], []

        for aug_img, transform_info in augmentations:
            results = model.predict(
                aug_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False
            )
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                scores_arr = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()

                boxes_orig = self.reverse_transform(boxes, transform_info)
                all_boxes.append(boxes_orig)
                all_scores.append(scores_arr)
                all_classes.append(clss)

        # Fuse predictions
        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0)
            keep = greedy_nms_classwise(boxes, scores, classes, iou_thres=tta_iou)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        # Annotate
        img_out = image.copy()
        if len(boxes) > 0:
            annotator = Annotator(img_out, line_width=2, example=model.names)
            for box, score, cls_id in zip(boxes, scores, classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(
                    [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    f"{name} {score:.2f}",
                    color=colors(int(cls_id), bgr=True),
                )
            img_out = annotator.result()

        elapsed = time.time() - t0
        return InferenceResult(
            img_out,
            boxes,
            scores,
            classes,
            self.name,
            params,
            elapsed,
            len(boxes),
        )

