"""
Hybrid pipeline: full-image inference + crops with simple smart fusion.
"""

import time

import numpy as np
from ultralytics.utils.plotting import Annotator, colors

from alpha5_base_inference import InferenceMethod, InferenceResult
from crop_utils import UniformCrops
from wbf_utils import weighted_boxes_fusion


class HybridInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "Hybrid",
            "Hybrid pipeline: full image + crops with simplified smart filtering",
        )
        self.default_params = {
            "conf": 0.25,
            "crops": 6,
            "overlap": 0.2,
            "crops_iou": 0.5,
            "high_iou": 0.85,
            "suspect_iou": 0.3,
            "merge_iou": 0.5,
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)
        crops_number = params.get("crops", 6)
        overlap = params.get("overlap", 0.2)
        crops_iou = params.get("crops_iou", 0.5)
        merge_iou = params.get("merge_iou", 0.5)

        # Stage 1: Full image
        results_full = model.predict(image, conf=conf, verbose=False)
        r = results_full[0]
        full_boxes, full_scores, full_classes = np.array([]), np.array([]), np.array([])
        if r.boxes is not None and len(r.boxes) > 0:
            full_boxes = r.boxes.xyxy.cpu().numpy()
            full_scores = r.boxes.conf.cpu().numpy()
            full_classes = r.boxes.cls.cpu().numpy()

        # Stage 2: Crops
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

        crops_boxes, crops_scores, crops_classes = np.array([]), np.array([]), np.array([])
        if all_boxes:
            crops_boxes = np.array(all_boxes, dtype=np.float32)
            crops_scores = np.array(all_scores, dtype=np.float32)
            crops_classes = np.array(all_classes, dtype=np.int32)
            crops_boxes, crops_scores, crops_classes = weighted_boxes_fusion(
                crops_boxes,
                crops_scores,
                crops_classes,
                iou_thres=crops_iou,
                skip_box_thr=conf,
            )

        # Stage 3: Simple merge (no complex suspicious/high IoU logic)
        if len(full_boxes) > 0 and len(crops_boxes) > 0:
            boxes = np.concatenate([full_boxes, crops_boxes], axis=0)
            scores = np.concatenate([full_scores, crops_scores], axis=0)
            classes = np.concatenate([full_classes, crops_classes], axis=0)
        elif len(full_boxes) > 0:
            boxes, scores, classes = full_boxes, full_scores, full_classes
        elif len(crops_boxes) > 0:
            boxes, scores, classes = crops_boxes, crops_scores, crops_classes
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        if len(boxes) > 0:
            boxes, scores, classes = weighted_boxes_fusion(
                boxes, scores, classes, iou_thres=merge_iou, skip_box_thr=0.0
            )

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

