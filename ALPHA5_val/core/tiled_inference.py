"""
Tiled inference method (uniform crops) with WBF/NMS fusion.
"""

import time

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

from alpha5_base_inference import InferenceMethod, InferenceResult
from crop_utils import UniformCrops
from wbf_utils import weighted_boxes_fusion, greedy_nms_classwise


class TiledInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "Tiled",
            "Tiled inference with uniform crops and WBF/NMS fusion",
        )
        self.default_params = {
            "conf": 0.25,
            "iou": 0.5,
            "crops": 4,
            "overlap": 0.2,
            "fusion": "wbf",
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)
        iou = params.get("iou", 0.5)
        crops_number = params.get("crops", 4)
        overlap = params.get("overlap", 0.2)
        fusion_method = params.get("fusion", "wbf")

        cropper = UniformCrops(overlap_ratio=overlap)
        crops, coords = cropper.crop(image, crops_number=crops_number)

        all_boxes, all_scores, all_classes = [], [], []

        for crop, (x_min, y_min, _, _) in zip(crops, coords):
            results = model.predict(crop, conf=conf, verbose=False)
            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores_arr = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            # Transform to global coordinates
            for b, s, c in zip(boxes, scores_arr, clss):
                all_boxes.append(
                    [
                        float(b[0] + x_min),
                        float(b[1] + y_min),
                        float(b[2] + x_min),
                        float(b[3] + y_min),
                    ]
                )
                all_scores.append(float(s))
                all_classes.append(int(c))

        # Fusion
        if all_boxes:
            boxes = np.array(all_boxes, dtype=np.float32)
            scores = np.array(all_scores, dtype=np.float32)
            classes = np.array(all_classes, dtype=np.int32)

            if fusion_method == "wbf":
                boxes, scores, classes = weighted_boxes_fusion(
                    boxes, scores, classes, iou_thres=iou, skip_box_thr=conf
                )
            else:  # nms
                keep = greedy_nms_classwise(boxes, scores, classes, iou_thres=iou)
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
                    box,
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

