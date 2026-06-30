"""
Utility functions for bounding box fusion and deduplication.
"""

import numpy as np


def compute_iou_xyxy(a, b) -> float:
    """Calculate IoU between two bounding boxes in xyxy format."""
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
    """Weighted Boxes Fusion with BFS clustering."""
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
                    if not visited[j] and compute_iou_xyxy(cls_boxes[current], cls_boxes[j]) >= iou_thres:
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
                np.sum(cluster_boxes[:, 3] * weights),
            ]

            fused_boxes.append(fused_box)
            fused_scores.append(np.max(cluster_scores))
            fused_classes.append(cls_id)

    if len(fused_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(fused_boxes, dtype=np.float32),
        np.array(fused_scores, dtype=np.float32),
        np.array(fused_classes, dtype=np.int32),
    )


def greedy_nms_classwise(boxes, scores, classes, iou_thres=0.5):
    """Per-class Non-Maximum Suppression."""
    keep = []
    for cls_id in sorted(set(classes)):
        idxs = [i for i, c in enumerate(classes) if c == cls_id]
        idxs.sort(key=lambda i: scores[i], reverse=True)

        picked = []
        for i in idxs:
            ok = True
            for j in picked:
                if compute_iou_xyxy(boxes[i], boxes[j]) > iou_thres:
                    ok = False
                    break
            if ok:
                picked.append(i)
        keep.extend(picked)

    keep.sort(key=lambda i: scores[i], reverse=True)
    return keep


def deduplicate_detections(boxes, scores, classes,
                           iou_threshold=0.5,
                           trash_class_id=7,
                           prioritize_non_trash=True,
                           keep_all=False):
    """
    Remove duplicate detections with class prioritization logic.

    If prioritize_non_trash=True, specific classes take priority over the
    generic 'trash' class when overlapping detections are found.

    Args:
        boxes: Array [N, 4] in xyxy format
        scores: Array [N] confidences
        classes: Array [N] class IDs
        iou_threshold: IoU >= this -> consider duplicates
        trash_class_id: ID of the generic class (default: 7)
        prioritize_non_trash: If True, specific classes beat trash on overlap
        keep_all: If True, skip deduplication and return all detections

    Returns:
        (boxes, scores, classes) filtered without duplicates
    """
    if keep_all or len(boxes) == 0:
        return boxes, scores, classes

    n = len(boxes)
    keep_mask = np.ones(n, dtype=bool)
    sorted_indices = np.argsort(-scores)

    for i in range(n):
        if not keep_mask[sorted_indices[i]]:
            continue

        idx_i = sorted_indices[i]
        box_i = boxes[idx_i]
        cls_i = classes[idx_i]

        for j in range(i + 1, n):
            idx_j = sorted_indices[j]
            if not keep_mask[idx_j]:
                continue

            box_j = boxes[idx_j]
            cls_j = classes[idx_j]

            if compute_iou_xyxy(box_i, box_j) >= iou_threshold:
                if prioritize_non_trash:
                    if cls_i == trash_class_id and cls_j != trash_class_id:
                        keep_mask[idx_i] = False
                        break
                    elif cls_i != trash_class_id and cls_j == trash_class_id:
                        keep_mask[idx_j] = False
                    else:
                        keep_mask[idx_j] = False
                else:
                    keep_mask[idx_j] = False

    return boxes[keep_mask], scores[keep_mask], classes[keep_mask]
