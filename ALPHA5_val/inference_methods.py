"""
Alpha5 Inference Methods - Unified Interface
Todos los métodos de inferencia del proyecto Alpha5 con interfaz común
"""
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Imports de tus módulos (deberán estar disponibles)
try:
    from patched_yolo_infer import MakeCropsDetectThem, CombineDetections
    PATCHED_AVAILABLE = True
except ImportError:
    PATCHED_AVAILABLE = False

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

from crop_utils import UniformCrops
from wbf_utils import weighted_boxes_fusion, greedy_nms_classwise, deduplicate_detections


class InferenceResult:
    """Resultado de una inferencia con metadatos"""
    def __init__(self, image, boxes, scores, classes, method_name, params, elapsed_time, num_detections):
        self.image = image  # Imagen anotada
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.method_name = method_name
        self.params = params
        self.elapsed_time = elapsed_time
        self.num_detections = num_detections


class InferenceMethod:
    """Clase base para todos los métodos de inferencia"""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.default_params = {}

    def run(self, image, model, params):
        """
        Ejecuta la inferencia en la imagen

        Args:
            image: numpy array (BGR) de la imagen
            model: YOLO model instance
            params: dict con parámetros específicos del método

        Returns:
            InferenceResult con imagen anotada y metadatos
        """
        raise NotImplementedError

    def get_params_config(self):
        """Retorna configuración de parámetros para la GUI"""
        return self.default_params


# ============= MÉTODO 1: Inferencia Básica =============
class BasicInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "Basic",
            "Inferencia básica YOLO sin modificaciones"
        )
        self.default_params = {
            'conf': 0.25,
            'iou': 0.45,
            'imgsz': 640
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get('conf', 0.25)
        iou = params.get('iou', 0.45)
        imgsz = params.get('imgsz', 640)

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
            img_out, boxes, scores, classes,
            self.name, params, elapsed, len(boxes)
        )


# ============= MÉTODO 2: Tiled Inference =============
class TiledInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "Tiled",
            "Tiled inference con crops uniformes y WBF/NMS"
        )
        self.default_params = {
            'conf': 0.25,
            'iou': 0.5,
            'crops': 4,
            'overlap': 0.2,
            'fusion': 'wbf'
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get('conf', 0.25)
        iou = params.get('iou', 0.5)
        crops_number = params.get('crops', 4)
        overlap = params.get('overlap', 0.2)
        fusion_method = params.get('fusion', 'wbf')

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

            # Transformar a coordenadas globales
            for b, s, c in zip(boxes, scores_arr, clss):
                all_boxes.append([float(b[0] + x_min), float(b[1] + y_min),
                                float(b[2] + x_min), float(b[3] + y_min)])
                all_scores.append(float(s))
                all_classes.append(int(c))

        # Fusión
        if all_boxes:
            boxes = np.array(all_boxes, dtype=np.float32)
            scores = np.array(all_scores, dtype=np.float32)
            classes = np.array(all_classes, dtype=np.int32)

            if fusion_method == 'wbf':
                boxes, scores, classes = weighted_boxes_fusion(
                    boxes, scores, classes, iou_thres=iou, skip_box_thr=conf
                )
            else:  # nms
                keep = greedy_nms_classwise(boxes, scores, classes, iou_thres=iou)
                boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        # Anotar
        img_out = image.copy()
        if len(boxes) > 0:
            annotator = Annotator(img_out, line_width=2, example=model.names)
            for box, score, cls_id in zip(boxes, scores, classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(box, f"{name} {score:.2f}", 
                                  color=colors(int(cls_id), bgr=True))
            img_out = annotator.result()

        elapsed = time.time() - t0
        return InferenceResult(
            img_out, boxes, scores, classes,
            self.name, params, elapsed, len(boxes)
        )


# ============= MÉTODO 3: Multi-Scale Ensemble =============
class MultiScaleInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "MultiScale",
            "Ensemble multi-escala con fusión NMS"
        )
        self.default_params = {
            'conf': 0.25,
            'iou': 0.5,
            'scales': [640, 960, 1280],
            'nms_thresh': 0.5
        }

    def run(self, image, model, params):
        t0 = time.time()
        conf = params.get('conf', 0.25)
        iou = params.get('iou', 0.5)
        scales = params.get('scales', [640, 960, 1280])
        nms_thresh = params.get('nms_thresh', 0.5)

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

        # Fusionar
        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0)
            keep = greedy_nms_classwise(boxes, scores, classes, iou_thres=nms_thresh)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        # Anotar
        img_out = image.copy()
        if len(boxes) > 0:
            annotator = Annotator(img_out, line_width=2, example=model.names)
            for box, score, cls_id in zip(boxes, scores, classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(
                    [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    f"{name} {score:.2f}",
                    color=colors(int(cls_id), bgr=True)
                )
            img_out = annotator.result()

        elapsed = time.time() - t0
        return InferenceResult(
            img_out, boxes, scores, classes,
            self.name, params, elapsed, len(boxes)
        )


# ============= MÉTODO 4: Test-Time Augmentation =============
class TTAInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "TTA",
            "Test-Time Augmentation con 5 transformaciones"
        )
        self.default_params = {
            'conf': 0.25,
            'iou': 0.5,
            'tta_iou': 0.5,
            'imgsz': 640
        }

    def apply_tta(self, image):
        h, w = image.shape[:2]
        augs = [
            (image.copy(), {"type": "original", "params": None}),
            (cv2.flip(image, 1), {"type": "flip_h", "params": {"width": w}}),
            (cv2.flip(image, 0), {"type": "flip_v", "params": {"height": h}}),
            (cv2.flip(image, -1), {"type": "flip_hv", "params": {"width": w, "height": h}}),
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
        conf = params.get('conf', 0.25)
        iou = params.get('iou', 0.5)
        imgsz = params.get('imgsz', 640)
        tta_iou = params.get('tta_iou', 0.5)

        augmentations = self.apply_tta(image)
        all_boxes, all_scores, all_classes = [], [], []

        for aug_img, transform_info in augmentations:
            results = model.predict(aug_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
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

        # Fusionar
        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0)
            keep = greedy_nms_classwise(boxes, scores, classes, iou_thres=tta_iou)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        # Anotar
        img_out = image.copy()
        if len(boxes) > 0:
            annotator = Annotator(img_out, line_width=2, example=model.names)
            for box, score, cls_id in zip(boxes, scores, classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(
                    [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    f"{name} {score:.2f}",
                    color=colors(int(cls_id), bgr=True)
                )
            img_out = annotator.result()

        elapsed = time.time() - t0
        return InferenceResult(
            img_out, boxes, scores, classes,
            self.name, params, elapsed, len(boxes)
        )


# ============= MÉTODO 5: Super Resolution =============
class SuperResolutionInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "SuperRes",
            "Preprocessing con CLAHE o Unsharp Mask"
        )
        self.default_params = {
            'conf': 0.25,
            'iou': 0.5,
            'imgsz': 640,
            'sr_method': 'clahe'
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
        conf = params.get('conf', 0.25)
        iou = params.get('iou', 0.5)
        imgsz = params.get('imgsz', 640)
        sr_method = params.get('sr_method', 'clahe')

        if sr_method == 'clahe':
            img_processed = self.apply_clahe(image)
        elif sr_method == 'unsharp':
            img_processed = self.apply_unsharp(image)
        else:
            img_processed = image

        results = model.predict(img_processed, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        r = results[0]

        boxes, scores, classes = [], [], []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

        img_out = r.plot()
        elapsed = time.time() - t0

        return InferenceResult(
            img_out, boxes, scores, classes,
            self.name, params, elapsed, len(boxes)
        )


# ============= MÉTODO 6: Hybrid Pipeline =============
class HybridInference(InferenceMethod):
    def __init__(self):
        super().__init__(
            "Hybrid",
            "Pipeline híbrido: Full + Crops con filtrado inteligente"
        )
        self.default_params = {
            'conf': 0.25,
            'crops': 6,
            'overlap': 0.2,
            'crops_iou': 0.5,
            'high_iou': 0.85,
            'suspect_iou': 0.3,
            'merge_iou': 0.5
        }

    def run(self, image, model, params):
        from wbf_utils import compute_iou_xyxy

        t0 = time.time()
        conf = params.get('conf', 0.25)
        crops_number = params.get('crops', 6)
        overlap = params.get('overlap', 0.2)
        crops_iou = params.get('crops_iou', 0.5)
        high_iou = params.get('high_iou', 0.85)
        suspect_iou = params.get('suspect_iou', 0.3)
        merge_iou = params.get('merge_iou', 0.5)

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
                all_boxes.append([float(b[0] + x_min), float(b[1] + y_min),
                                float(b[2] + x_min), float(b[3] + y_min)])
                all_scores.append(float(s))
                all_classes.append(int(c))

        crops_boxes, crops_scores, crops_classes = np.array([]), np.array([]), np.array([])
        if all_boxes:
            crops_boxes = np.array(all_boxes, dtype=np.float32)
            crops_scores = np.array(all_scores, dtype=np.float32)
            crops_classes = np.array(all_classes, dtype=np.int32)
            crops_boxes, crops_scores, crops_classes = weighted_boxes_fusion(
                crops_boxes, crops_scores, crops_classes,
                iou_thres=crops_iou, skip_box_thr=conf
            )

        # Stage 3: Smart filter (simplificado)
        # Merge directamente
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

        # Anotar
        img_out = image.copy()
        if len(boxes) > 0:
            annotator = Annotator(img_out, line_width=2, example=model.names)
            for box, score, cls_id in zip(boxes, scores, classes):
                name = model.names.get(int(cls_id), str(int(cls_id)))
                annotator.box_label(box, f"{name} {score:.2f}", 
                                  color=colors(int(cls_id), bgr=True))
            img_out = annotator.result()

        elapsed = time.time() - t0
        return InferenceResult(
            img_out, boxes, scores, classes,
            self.name, params, elapsed, len(boxes)
        )


# ============= REGISTRO DE MÉTODOS =============
AVAILABLE_METHODS = {
    'basic': BasicInference(),
    'tiled': TiledInference(),
    'multiscale': MultiScaleInference(),
    'tta': TTAInference(),
    'superres': SuperResolutionInference(),
    'hybrid': HybridInference(),
}

def get_available_methods():
    """Retorna lista de métodos disponibles"""
    return list(AVAILABLE_METHODS.keys())

def get_method(method_name):
    """Obtiene un método por nombre"""
    return AVAILABLE_METHODS.get(method_name)
