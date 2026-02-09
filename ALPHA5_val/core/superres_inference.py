"""
Super-resolution-style preprocessing inference method (CLAHE / Unsharp).
"""

import time

import cv2

from alpha5_base_inference import InferenceMethod, InferenceResult


class SuperResolutionInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "SuperRes",
            "Preprocessing with CLAHE or Unsharp Mask",
        )
        self.default_params = {
            "conf": 0.25,
            "iou": 0.5,
            "imgsz": 640,
            "sr_method": "clahe",
        }

    def apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def apply_unsharp(self, image, kernel_size=5, strength=1.5):
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 1.0)
        sharpened = cv2.addWeighted(image, strength, blurred, -(strength - 1), 0)
        return sharpened

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)
        iou = params.get("iou", 0.5)
        imgsz = params.get("imgsz", 640)
        sr_method = params.get("sr_method", "clahe")

        if sr_method == "clahe":
            img_processed = self.apply_clahe(image)
        elif sr_method == "unsharp":
            img_processed = self.apply_unsharp(image)
        else:
            img_processed = image

        results = model.predict(
            img_processed, conf=conf, iou=iou, imgsz=imgsz, verbose=False
        )
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

