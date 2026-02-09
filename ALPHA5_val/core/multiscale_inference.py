"""
Multi-scale ensemble inference method with NMS fusion.
"""

import time

import numpy as np
from ultralytics.utils.plotting import Annotator, colors

from alpha5_base_inference import InferenceMethod, InferenceResult
from wbf_utils import greedy_nms_classwise


class MultiScaleInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "MultiScale",
            "Multi-scale ensemble with NMS fusion",
        )
        self.default_params = {
            "conf": 0.25,
            "iou": 0.5,
            "scales": [640, 960, 1280],
            "nms_thresh": 0.5,
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)
        iou = params.get("iou", 0.5)
        scales = params.get("scales", [640, 960, 1280])
        nms_thresh = params.get("nms_thresh", 0.5)

        all_boxes, all_scores, all_classes = [], [], []

        for scale in scales:
            results = model.predict(image, conf=conf, iou=iou, imgsz=scale, verbose=False)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                scores_arr = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                all_boxes.append(boxes)
                all_scores.append(scores_arr)
                all_classes.append(clss)

        # Fuse predictions
        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0)
            keep = greedy_nms_classwise(boxes, scores, classes, iou_thres=nms_thresh)
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

