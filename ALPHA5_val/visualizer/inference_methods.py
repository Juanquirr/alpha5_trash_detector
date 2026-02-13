"""
Alpha5 Inference Methods with deduplication and class prioritization support

Enhanced version with:
- SuperResolution: Configurable method (CLAHE, Unsharp, Both)
- TTA: Enhanced with rotation and scale augmentations
"""

import time
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from alpha5_base import InferenceMethod, InferenceResult
from utils import UniformCrops, weighted_boxes_fusion, greedy_nms, deduplicate_detections


# ============= BASIC =============

class BasicInference(InferenceMethod):
    def __init__(self):
        super().__init__("Basic", "Basic YOLO inference")
        self.default_params = {
            "conf": 0.25,
            "iou": 0.45,
            "imgsz": 640,
            "deduplicate": False,
            "dedup_iou": 0.5,
            "trash_class_id": 7,
            "prioritize_specific": True
        }

    def run(self, image, model, params):
        t0 = time.time()

        results = model.predict(image,
                              conf=params.get("conf", 0.25),
                              iou=params.get("iou", 0.45),
                              imgsz=params.get("imgsz", 640),
                              verbose=False)

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.array([])
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.array([])
        classes = r.boxes.cls.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.array([])

        # Optional deduplication
        if params.get("deduplicate", False) and len(boxes) > 0:
            boxes, scores, classes = deduplicate_detections(
                boxes, scores, classes,
                iou_threshold=params.get("dedup_iou", 0.5),
                trash_class_id=params.get("trash_class_id", 7),
                prioritize_specific=params.get("prioritize_specific", True)
            )

        img_out = r.plot()
        elapsed = time.time() - t0

        return InferenceResult(img_out, boxes, scores, classes, self.name, params, elapsed, len(boxes))


# ============= TILED =============

class TiledInference(InferenceMethod):
    def __init__(self):
        super().__init__("Tiled", "Inference with crops and fusion")
        self.default_params = {
            "conf": 0.25,
            "iou": 0.5,
            "crops": 4,
            "overlap": 0.2,
            "fusion": "wbf",
            "deduplicate": True,  # Enabled by default in Tiled
            "dedup_iou": 0.5,
            "trash_class_id": 7,
            "prioritize_specific": True
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)

        cropper = UniformCrops(overlap_ratio=params.get("overlap", 0.2))
        crops, coords = cropper.crop(image, crops_number=params.get("crops", 4))

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
                all_boxes.append([b[0]+x_min, b[1]+y_min, b[2]+x_min, b[3]+y_min])
                all_scores.append(s)
                all_classes.append(c)

        if all_boxes:
            boxes = np.array(all_boxes, dtype=np.float32)
            scores = np.array(all_scores, dtype=np.float32)
            classes = np.array(all_classes, dtype=np.int32)

            if params.get("fusion", "wbf") == "wbf":
                boxes, scores, classes = weighted_boxes_fusion(boxes, scores, classes,
                                                              iou_thres=params.get("iou", 0.5))
            else:
                keep = greedy_nms(boxes, scores, classes, iou_thres=params.get("iou", 0.5))
                boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

            # Post-fusion deduplication
            if params.get("deduplicate", True):
                boxes, scores, classes = deduplicate_detections(
                    boxes, scores, classes,
                    iou_threshold=params.get("dedup_iou", 0.5),
                    trash_class_id=params.get("trash_class_id", 7),
                    prioritize_specific=params.get("prioritize_specific", True)
                )
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        img_out = image.copy()
        if len(boxes) > 0:
            annotator = Annotator(img_out, line_width=2, example=model.names)
            for box, score, cls_id in zip(boxes, scores, classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(box, f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
            img_out = annotator.result()

        elapsed = time.time() - t0
        return InferenceResult(img_out, boxes, scores, classes, self.name, params, elapsed, len(boxes))


# ============= MULTISCALE =============

class MultiScaleInference(InferenceMethod):
    def __init__(self):
        super().__init__("MultiScale", "Multi-scale ensemble")
        self.default_params = {
            "conf": 0.25,
            "iou": 0.5,
            "scales": [640, 960, 1280],
            "nms_thresh": 0.5,
            "deduplicate": True,
            "dedup_iou": 0.5,
            "trash_class_id": 7,
            "prioritize_specific": True
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)
        iou = params.get("iou", 0.5)
        scales = params.get("scales", [640, 960, 1280])

        all_boxes, all_scores, all_classes = [], [], []

        for scale in scales:
            results = model.predict(image, conf=conf, iou=iou, imgsz=scale, verbose=False)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                all_boxes.append(r.boxes.xyxy.cpu().numpy())
                all_scores.append(r.boxes.conf.cpu().numpy())
                all_classes.append(r.boxes.cls.cpu().numpy())

        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0)

            keep = greedy_nms(boxes, scores, classes, iou_thres=params.get("nms_thresh", 0.5))
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

            # Deduplication
            if params.get("deduplicate", True):
                boxes, scores, classes = deduplicate_detections(
                    boxes, scores, classes,
                    iou_threshold=params.get("dedup_iou", 0.5),
                    trash_class_id=params.get("trash_class_id", 7),
                    prioritize_specific=params.get("prioritize_specific", True)
                )
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        img_out = image.copy()
        if len(boxes) > 0:
            annotator = Annotator(img_out, line_width=2, example=model.names)
            for box, score, cls_id in zip(boxes, scores, classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label([float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                                   f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
            img_out = annotator.result()

        elapsed = time.time() - t0
        return InferenceResult(img_out, boxes, scores, classes, self.name, params, elapsed, len(boxes))


# ============= TTA (ENHANCED) =============

class TTAInference(InferenceMethod):
    def __init__(self):
        super().__init__("TTA", "Enhanced Test-Time Augmentation")
        self.default_params = {
            "conf": 0.25,
            "iou": 0.5,
            "tta_iou": 0.5,
            "imgsz": 640,
            "use_flips": True,
            "use_brightness": False,  # Optional brightness augmentation
            "deduplicate": True,
            "dedup_iou": 0.5,
            "trash_class_id": 7,
            "prioritize_specific": True
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)
        iou = params.get("iou", 0.5)
        imgsz = params.get("imgsz", 640)
        h, w = image.shape[:2]

        # Build augmentations list
        augmentations = [
            (image.copy(), None)  # Original
        ]

        if params.get("use_flips", True):
            augmentations.extend([
                (cv2.flip(image, 1), "flip_h"),
                (cv2.flip(image, 0), "flip_v"),
                (cv2.flip(image, -1), "flip_hv")
            ])

        # Optional: brightness augmentation
        if params.get("use_brightness", False):
            bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
            dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
            augmentations.extend([
                (bright, "bright"),
                (dark, "dark")
            ])

        all_boxes, all_scores, all_classes = [], [], []

        for aug_img, transform in augmentations:
            results = model.predict(aug_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()

                # Reverse transformations
                if transform == "flip_h":
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                elif transform == "flip_v":
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                elif transform == "flip_hv":
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                # Brightness transforms don't need box adjustment

                all_boxes.append(boxes)
                all_scores.append(r.boxes.conf.cpu().numpy())
                all_classes.append(r.boxes.cls.cpu().numpy())

        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0)

            keep = greedy_nms(boxes, scores, classes, iou_thres=params.get("tta_iou", 0.5))
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

            # Deduplication
            if params.get("deduplicate", True):
                boxes, scores, classes = deduplicate_detections(
                    boxes, scores, classes,
                    iou_threshold=params.get("dedup_iou", 0.5),
                    trash_class_id=params.get("trash_class_id", 7),
                    prioritize_specific=params.get("prioritize_specific", True)
                )
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        img_out = image.copy()
        if len(boxes) > 0:
            annotator = Annotator(img_out, line_width=2, example=model.names)
            for box, score, cls_id in zip(boxes, scores, classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label([float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                                   f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
            img_out = annotator.result()

        elapsed = time.time() - t0
        return InferenceResult(img_out, boxes, scores, classes, self.name, params, elapsed, len(boxes))


# ============= SUPER RESOLUTION (ENHANCED) =============

class SuperResInference(InferenceMethod):
    def __init__(self):
        super().__init__("SuperRes", "Enhanced preprocessing: CLAHE/Unsharp/Both")
        self.default_params = {
            "conf": 0.25,
            "iou": 0.5,
            "imgsz": 640,
            "sr_method": "clahe",  # Options: 'clahe', 'unsharp', 'both'
            "clahe_clip": 3.0,
            "clahe_tile": 8,
            "unsharp_sigma": 1.0,
            "unsharp_strength": 1.5,
            "deduplicate": False,
            "dedup_iou": 0.5,
            "trash_class_id": 7,
            "prioritize_specific": True
        }

    def run(self, image, model, params):
        t0 = time.time()

        sr_method = params.get("sr_method", "clahe")
        img_processed = image.copy()

        # Apply preprocessing based on method
        if sr_method == "clahe":
            img_processed = self._apply_clahe(img_processed, params)
        elif sr_method == "unsharp":
            img_processed = self._apply_unsharp(img_processed, params)
        elif sr_method == "both":
            # Apply CLAHE first, then unsharp mask
            img_processed = self._apply_clahe(img_processed, params)
            img_processed = self._apply_unsharp(img_processed, params)

        # Run inference on processed image
        results = model.predict(img_processed,
                              conf=params.get("conf", 0.25),
                              iou=params.get("iou", 0.5),
                              imgsz=params.get("imgsz", 640),
                              verbose=False)

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.array([])
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.array([])
        classes = r.boxes.cls.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.array([])

        # Optional deduplication
        if params.get("deduplicate", False) and len(boxes) > 0:
            boxes, scores, classes = deduplicate_detections(
                boxes, scores, classes,
                iou_threshold=params.get("dedup_iou", 0.5),
                trash_class_id=params.get("trash_class_id", 7),
                prioritize_specific=params.get("prioritize_specific", True)
            )

        img_out = r.plot()
        elapsed = time.time() - t0

        return InferenceResult(img_out, boxes, scores, classes, self.name, params, elapsed, len(boxes))

    def _apply_clahe(self, image, params):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clip_limit = params.get("clahe_clip", 3.0)
        tile_size = params.get("clahe_tile", 8)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)

        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _apply_unsharp(self, image, params):
        """Apply Unsharp Mask for image sharpening"""
        sigma = params.get("unsharp_sigma", 1.0)
        strength = params.get("unsharp_strength", 1.5)

        blurred = cv2.GaussianBlur(image, (5, 5), sigma)
        return cv2.addWeighted(image, strength, blurred, -(strength - 1.0), 0)


# ============= HYBRID =============

class HybridInference(InferenceMethod):
    def __init__(self):
        super().__init__("Hybrid", "Full image + crops")
        self.default_params = {
            "conf": 0.25,
            "crops": 6,
            "overlap": 0.2,
            "merge_iou": 0.5,
            "deduplicate": True,  # Enabled by default
            "dedup_iou": 0.5,
            "trash_class_id": 7,
            "prioritize_specific": True
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get("conf", 0.25)

        # Full image detection
        results_full = model.predict(image, conf=conf, verbose=False)
        r = results_full[0]
        full_boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.array([])
        full_scores = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.array([])
        full_classes = r.boxes.cls.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.array([])

        # Crops detection
        cropper = UniformCrops(overlap_ratio=params.get("overlap", 0.2))
        crops, coords = cropper.crop(image, crops_number=params.get("crops", 6))

        all_boxes, all_scores, all_classes = [], [], []

        for crop, (x_min, y_min, _, _) in zip(crops, coords):
            results = model.predict(crop, conf=conf, verbose=False)
            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            for b, s, c in zip(boxes, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                all_boxes.append([b[0]+x_min, b[1]+y_min, b[2]+x_min, b[3]+y_min])
                all_scores.append(s)
                all_classes.append(c)

        crops_boxes = np.array(all_boxes, dtype=np.float32) if all_boxes else np.array([])
        crops_scores = np.array(all_scores, dtype=np.float32) if all_boxes else np.array([])
        crops_classes = np.array(all_classes, dtype=np.int32) if all_boxes else np.array([])

        # Merge full and crops
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
            boxes, scores, classes = weighted_boxes_fusion(boxes, scores, classes,
                                                          iou_thres=params.get("merge_iou", 0.5))

            # Post-merge deduplication
            if params.get("deduplicate", True):
                boxes, scores, classes = deduplicate_detections(
                    boxes, scores, classes,
                    iou_threshold=params.get("dedup_iou", 0.5),
                    trash_class_id=params.get("trash_class_id", 7),
                    prioritize_specific=params.get("prioritize_specific", True)
                )

        img_out = image.copy()
        if len(boxes) > 0:
            annotator = Annotator(img_out, line_width=2, example=model.names)
            for box, score, cls_id in zip(boxes, scores, classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(box, f"{name} {score:.2f}", color=colors(int(cls_id), bgr=True))
            img_out = annotator.result()

        elapsed = time.time() - t0
        return InferenceResult(img_out, boxes, scores, classes, self.name, params, elapsed, len(boxes))


# ============= REGISTRY =============

METHODS = {
    "basic": BasicInference(),
    "tiled": TiledInference(),
    "multiscale": MultiScaleInference(),
    "tta": TTAInference(),
    "superres": SuperResInference(),
    "hybrid": HybridInference(),
}


def get_available_methods():
    return list(METHODS.keys())


def get_method(method_name):
    return METHODS.get(method_name)
