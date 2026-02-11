"""
Utilidades: Crops uniformes, WBF, NMS y Deduplicación con priorización de clases
"""
import numpy as np
import math
import cv2

# ============= CROP UTILS =============
class UniformCrops:
    def __init__(self, overlap_ratio=0.2):
        if not (0 <= overlap_ratio < 1):
            raise ValueError("overlap_ratio debe estar en [0, 1)")
        self.overlap_ratio = overlap_ratio

    def crop(self, frame, crops_number=4):
        if (crops_number % 2 != 0) or (crops_number <= 0):
            raise ValueError("crops_number debe ser par y positivo")

        height, width = frame.shape[:2]
        is_vertical = height > width

        base_rows = int(math.sqrt(crops_number))
        base_cols = math.ceil(crops_number / base_rows)

        if is_vertical:
            grid_rows, grid_cols = base_cols, base_rows
        else:
            grid_rows, grid_cols = base_rows, base_cols

        cell_w = width / (grid_cols - (grid_cols - 1) * self.overlap_ratio)
        cell_h = height / (grid_rows - (grid_rows - 1) * self.overlap_ratio)
        stride_w = cell_w * (1 - self.overlap_ratio)
        stride_h = cell_h * (1 - self.overlap_ratio)

        crops, coords = [], []
        count = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                if count >= crops_number:
                    break
                x_min = max(0.0, c * stride_w)
                y_min = max(0.0, r * stride_h)
                x_max = min(float(width), x_min + cell_w)
                y_max = min(float(height), y_min + cell_h)

                x1, y1 = int(round(x_min)), int(round(y_min))
                x2, y2 = int(round(x_max)), int(round(y_max))

                crops.append(frame[y1:y2, x1:x2])
                coords.append((x_min, y_min, x_max, y_max))
                count += 1

        return crops, coords

# ============= WBF & NMS =============
def compute_iou(a, b):
    """Calcula IoU entre dos bounding boxes en formato xyxy"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def weighted_boxes_fusion(boxes, scores, classes, iou_thres=0.5, skip_box_thr=0.0):
    """Weighted Boxes Fusion con clustering BFS"""
    if len(boxes) == 0:
        return boxes, scores, classes

    fused_boxes, fused_scores, fused_classes = [], [], []

    for cls_id in sorted(set(classes)):
        cls_mask = classes == cls_id
        cls_boxes = boxes[cls_mask].copy()
        cls_scores = scores[cls_mask].copy()

        valid_mask = cls_scores >= skip_box_thr
        cls_boxes = cls_boxes[valid_mask]
        cls_scores = cls_scores[valid_mask]

        if len(cls_boxes) == 0:
            continue

        n = len(cls_boxes)
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue

            cluster = [i]
            queue = [i]
            visited[i] = True

            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if not visited[j] and compute_iou(cls_boxes[current], cls_boxes[j]) >= iou_thres:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)

            cluster_boxes = cls_boxes[cluster]
            cluster_scores = cls_scores[cluster]
            weights = cluster_scores / cluster_scores.sum()

            fused_box = [
                np.sum(cluster_boxes[:, 0] * weights),
                np.sum(cluster_boxes[:, 1] * weights),
                np.sum(cluster_boxes[:, 2] * weights),
                np.sum(cluster_boxes[:, 3] * weights)
            ]
            fused_boxes.append(fused_box)
            fused_scores.append(np.max(cluster_scores))
            fused_classes.append(cls_id)

    if len(fused_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(fused_boxes, dtype=np.float32),
        np.array(fused_scores, dtype=np.float32),
        np.array(fused_classes, dtype=np.int32)
    )

def greedy_nms(boxes, scores, classes, iou_thres=0.5):
    """NMS por clase (Non-Maximum Suppression)"""
    keep = []
    for cls_id in sorted(set(classes)):
        idxs = [i for i, c in enumerate(classes) if c == cls_id]
        idxs.sort(key=lambda i: scores[i], reverse=True)

        picked = []
        for i in idxs:
            ok = True
            for j in picked:
                if compute_iou(boxes[i], boxes[j]) > iou_thres:
                    ok = False
                    break
            if ok:
                picked.append(i)
        keep.extend(picked)

    keep.sort(key=lambda i: scores[i], reverse=True)
    return keep

# ============= DEDUPLICACIÓN CON PRIORIZACIÓN =============
def deduplicate_detections(boxes, scores, classes,
                          iou_threshold=0.5,
                          trash_class_id=7,
                          prioritize_specific=True):
    """
    Elimina detecciones duplicadas con lógica de priorización de clases.

    Si prioritize_specific=True:
        - Las clases específicas se priorizan sobre la clase genérica "trash"
        - Útil para sistemas de detección de residuos donde hay una clase
          genérica "trash" y clases específicas (plástico, papel, metal, etc.)

    Args:
        boxes: Array [N, 4] en formato xyxy
        scores: Array [N] confianzas
        classes: Array [N] IDs de clase
        iou_threshold: IoU >= esto → considerar duplicados
        trash_class_id: ID de la clase genérica (default: 7)
        prioritize_specific: Si True, priorizar clases específicas sobre trash

    Returns:
        (boxes, scores, classes) filtrados sin duplicados

    Ejemplo:
        Si hay dos detecciones solapadas:
        - Detección A: clase=7 (trash), conf=0.8
        - Detección B: clase=2 (plastic), conf=0.6

        Con prioritize_specific=True → se queda con B (clase específica)
        Con prioritize_specific=False → se queda con A (mayor confianza)
    """
    if len(boxes) == 0:
        return boxes, scores, classes

    n = len(boxes)
    keep_mask = np.ones(n, dtype=bool)

    # Ordenar por confianza descendente
    sorted_indices = np.argsort(-scores)

    for i in range(n):
        if not keep_mask[sorted_indices[i]]:
            continue

        idx_i = sorted_indices[i]
        box_i = boxes[idx_i]
        cls_i = classes[idx_i]

        # Buscar detecciones solapadas
        for j in range(i + 1, n):
            idx_j = sorted_indices[j]
            if not keep_mask[idx_j]:
                continue

            box_j = boxes[idx_j]
            cls_j = classes[idx_j]

            iou = compute_iou(box_i, box_j)

            if iou >= iou_threshold:
                # Detecciones solapadas encontradas
                if prioritize_specific:
                    # Lógica especial: trash pierde ante cualquier clase específica
                    if cls_i == trash_class_id and cls_j != trash_class_id:
                        # Mantener j (clase específica), descartar i (trash)
                        keep_mask[idx_i] = False
                        break  # i descartado, pasar al siguiente
                    elif cls_i != trash_class_id and cls_j == trash_class_id:
                        # Mantener i (clase específica), descartar j (trash)
                        keep_mask[idx_j] = False
                    else:
                        # Ambos misma prioridad: mantener mayor confianza (i)
                        keep_mask[idx_j] = False
                else:
                    # Estándar: mantener mayor confianza (siempre i por estar ordenado)
                    keep_mask[idx_j] = False

    return boxes[keep_mask], scores[keep_mask], classes[keep_mask]
