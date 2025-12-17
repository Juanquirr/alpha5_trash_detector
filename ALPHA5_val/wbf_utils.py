import numpy as np


def compute_iou_xyxy(a, b) -> float:
    """
    Compute Intersection over Union for two bounding boxes in xyxy format.
    
    Args:
        a: First box [x1, y1, x2, y2]
        b: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
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
    """
    Weighted Boxes Fusion using BFS clustering for overlapping detections.
    
    Args:
        boxes: Array of bounding boxes [N, 4] in format (x1, y1, x2, y2)
        scores: Array of confidence scores [N]
        classes: Array of class IDs [N]
        iou_thres: IoU threshold for fusion (boxes with IoU >= threshold are merged)
        skip_box_thr: Minimum confidence score to consider a box
        
    Returns:
        Tuple of (fused_boxes, fused_scores, fused_classes) as numpy arrays
    """
    if len(boxes) == 0:
        return boxes, scores, classes
    
    fused_boxes = []
    fused_scores = []
    fused_classes = []
    
    # Process each class separately
    for cls_id in sorted(set(classes)):
        cls_mask = classes == cls_id
        cls_boxes = boxes[cls_mask].copy()
        cls_scores = scores[cls_mask].copy()
        
        if len(cls_boxes) == 0:
            continue
        
        # Filter by minimum score
        valid_mask = cls_scores >= skip_box_thr
        cls_boxes = cls_boxes[valid_mask]
        cls_scores = cls_scores[valid_mask]
        
        if len(cls_boxes) == 0:
            continue
        
        # Build IoU matrix for clustering
        n = len(cls_boxes)
        iou_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                iou = compute_iou_xyxy(cls_boxes[i], cls_boxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        # Find connected components (clusters of overlapping boxes)
        visited = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if visited[i]:
                continue
            
            # BFS to find all connected boxes
            cluster = [i]
            queue = [i]
            visited[i] = True
            
            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if not visited[j] and iou_matrix[current, j] >= iou_thres:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
            
            # Fuse all boxes in cluster using weighted average
            cluster_boxes = cls_boxes[cluster]
            cluster_scores = cls_scores[cluster]
            
            # Use scores as weights
            weights = cluster_scores / cluster_scores.sum()
            
            # Weighted coordinates
            fused_x1 = np.sum(cluster_boxes[:, 0] * weights)
            fused_y1 = np.sum(cluster_boxes[:, 1] * weights)
            fused_x2 = np.sum(cluster_boxes[:, 2] * weights)
            fused_y2 = np.sum(cluster_boxes[:, 3] * weights)
            
            # Use maximum confidence
            fused_conf = np.max(cluster_scores)
            
            fused_boxes.append([fused_x1, fused_y1, fused_x2, fused_y2])
            fused_scores.append(fused_conf)
            fused_classes.append(cls_id)
    
    if len(fused_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return (
        np.array(fused_boxes, dtype=np.float32),
        np.array(fused_scores, dtype=np.float32),
        np.array(fused_classes, dtype=np.int32)
    )


def greedy_nms_classwise(boxes, scores, classes, iou_thres: float):
    """
    Apply class-wise Non-Maximum Suppression (classic method).
    
    Args:
        boxes: Array of bounding boxes [N, 4]
        scores: Array of confidence scores [N]
        classes: Array of class IDs [N]
        iou_thres: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
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
    Remove duplicate detections, keeping highest confidence.
    Special handling for 'trash' class: deprioritize in favor of specific classes.
    
    Args:
        boxes: Array [N, 4] in xyxy format
        scores: Array [N] confidence scores
        classes: Array [N] class IDs
        iou_threshold: IoU >= this → consider duplicates
        trash_class_id: Class ID for generic "trash" (default: 7)
        prioritize_non_trash: If True, prefer non-trash classes over trash
        keep_all: If True, skip deduplication (show all detections)
        
    Returns:
        Filtered (boxes, scores, classes) with duplicates removed
    """
    if len(boxes) == 0 or keep_all:
        return boxes, scores, classes
    
    n = len(boxes)
    keep_mask = np.ones(n, dtype=bool)
    
    # Sort by confidence descending
    sorted_indices = np.argsort(-scores)
    
    for i in range(n):
        if not keep_mask[sorted_indices[i]]:
            continue
        
        idx_i = sorted_indices[i]
        box_i = boxes[idx_i]
        score_i = scores[idx_i]
        cls_i = classes[idx_i]
        
        # Find overlapping detections
        for j in range(i + 1, n):
            idx_j = sorted_indices[j]
            
            if not keep_mask[idx_j]:
                continue
            
            box_j = boxes[idx_j]
            cls_j = classes[idx_j]
            
            iou = compute_iou_xyxy(box_i, box_j)
            
            if iou >= iou_threshold:
                # Overlapping detections found
                
                if prioritize_non_trash:
                    # Special logic: trash loses to any specific class
                    if cls_i == trash_class_id and cls_j != trash_class_id:
                        # Keep j (specific class), discard i (trash)
                        keep_mask[idx_i] = False
                        break  # i is discarded, move to next
                    elif cls_i != trash_class_id and cls_j == trash_class_id:
                        # Keep i (specific class), discard j (trash)
                        keep_mask[idx_j] = False
                    else:
                        # Both same priority: keep higher confidence (i)
                        keep_mask[idx_j] = False
                else:
                    # Standard: keep higher confidence (always i since sorted)
                    keep_mask[idx_j] = False
    
    return boxes[keep_mask], scores[keep_mask], classes[keep_mask]
