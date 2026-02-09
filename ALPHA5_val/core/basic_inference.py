"""
Basic YOLO inference method without any modifications.
"""

import time

from ultralytics.utils.plotting import Annotator

from alpha5_base_inference import InferenceMethod, InferenceResult


class BasicInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "Basic",
            "Basic YOLO inference without modifications",
        )
        self.default_params = {
            "conf": 0.25,
            "iou": 0.45,
            "imgsz": 640,
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)
        iou = params.get("iou", 0.45)
        imgsz = params.get("imgsz", 640)

        results = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        r = results[0]

        boxes, scores, classes = [], [], []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

        img_out = r.plot()
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

