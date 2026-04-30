"""Geometric operations: IoU, NMS (with class priority), mask→bbox, mask→YOLO."""

from __future__ import annotations

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def nms(
    instances: list[tuple[np.ndarray, float, str, int]],
    iou_threshold: float,
    priority_map: dict[int, int],
) -> list[tuple[np.ndarray, float, str, int]]:
    """
    NMS with class priority. Sorts by (priority ASC, score DESC).
    Lower priority number = more specific class = wins overlapping detections.
    """
    def _key(x: tuple) -> tuple:
        _, score, _, class_id = x
        return (priority_map.get(class_id, 99), -score)

    kept: list[tuple[np.ndarray, float, str, int]] = []
    for item in sorted(instances, key=_key):
        mask = item[0]
        if all(iou(mask, k[0]) < iou_threshold for k in kept):
            kept.append(item)
    return kept


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Returns (x0, y0, x1, y1) pixel bbox, or None if mask is empty."""
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def mask_to_yolo(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    """Returns (cx, cy, w, h) normalized [0, 1] for YOLO format, or None."""
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w  = (x1 - x0) / img_w
    h  = (y1 - y0) / img_h
    return cx, cy, w, h
