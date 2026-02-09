import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from wbf_utils import deduplicate_detections


def annotate_detections(img, boxes, scores, classes, model, title_suffix=""):
    """Helper to annotate image with bounding boxes."""
    img_out = img.copy()
    annotator = Annotator(img_out, line_width=2, example=model.names)
    
    for box, score, cls_id in zip(boxes, scores, classes):
        name = model.names.get(int(cls_id), str(int(cls_id)))
        label = f"{name} {score:.2f}"
        annotator.box_label(box, label, color=colors(int(cls_id), bgr=True))
    
    # Add text overlay
    if title_suffix:
        cv2.putText(img_out, title_suffix, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return annotator.result()


def test_image(model, img_path, out_dir, conf, device, iou_dedup, trash_id):
    """
    Run inference and save 3 versions:
    1. All detections (raw)
    2. Deduplicated (highest confidence wins)
    3. Deduplicated + trash deprioritized
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Cannot read: {img_path}")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    results = model.predict(img, device=device, conf=conf, verbose=False)
    r = results[0]
    
    if r.boxes is None or len(r.boxes) == 0:
        print(f"⚠️  No detections in: {img_path.name}")
        return
    
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    
    print(f"\n📸 {img_path.name}")
    print(f"   Raw detections: {len(boxes)}")
    
    # Version 1: All detections
    img1 = annotate_detections(img, boxes, scores, classes, model, "All detections")
    cv2.imwrite(str(out_dir / f"{img_path.stem}_1_all.jpg"), img1)
    
    # Version 2: Deduplicated (standard)
    boxes2, scores2, classes2 = deduplicate_detections(
        boxes, scores, classes,
        iou_threshold=iou_dedup,
        prioritize_non_trash=False,
        keep_all=False
    )
    print(f"   Deduplicated (standard): {len(boxes2)}")
    img2 = annotate_detections(img, boxes2, scores2, classes2, model, 
                               "Deduplicated (highest conf)")
    cv2.imwrite(str(out_dir / f"{img_path.stem}_2_dedup.jpg"), img2)
    
    # Version 3: Deduplicated + trash deprioritized
    boxes3, scores3, classes3 = deduplicate_detections(
        boxes, scores, classes,
        iou_threshold=iou_dedup,
        trash_class_id=trash_id,
        prioritize_non_trash=True,
        keep_all=False
    )
    print(f"   Deduplicated (trash deprioritized): {len(boxes3)}")
    
    # Count trash detections
    trash_count = np.sum(classes3 == trash_id)
    print(f"   → 'trash' detections in final: {trash_count}")
    
    img3 = annotate_detections(img, boxes3, scores3, classes3, model, 
                               "Deduplicated (trash deprioritized)")
    cv2.imwrite(str(out_dir / f"{img_path.stem}_3_dedup_smart.jpg"), img3)
    
    # Detailed report
    print(f"\n   Class distribution (version 3):")
    for cls_id in sorted(set(classes3)):
        cls_name = model.names.get(int(cls_id), str(int(cls_id)))
        count = np.sum(classes3 == cls_id)
        avg_conf = np.mean(scores3[classes3 == cls_id])
        print(f"     - {cls_name}: {count} detections (avg conf: {avg_conf:.2f})")


def build_args():
    p = argparse.ArgumentParser(
        description="Test deduplication logic with visual comparison"
    )
    p.add_argument("source", help="Input image path")
    p.add_argument("model", help="Path to YOLO weights (.pt)")
    
    p.add_argument("--out_dir", default="dedup_test_results", 
                   help="Output directory")
    p.add_argument("--conf", type=float, default=0.15, 
                   help="Lower confidence to get overlapping detections (try 0.15)")
    p.add_argument("--device", default="cuda:0", 
                   help="Device")
    p.add_argument("--iou_dedup", type=float, default=0.5,
                   help="IoU threshold to consider detections as duplicates")
    p.add_argument("--trash_id", type=int, default=7,
                   help="Class ID for 'trash' (default: 7)")
    
    return p.parse_args()


def main():
    args = build_args()
    
    img_path = Path(args.source)
    out_dir = Path(args.out_dir)
    
    if not img_path.exists():
        raise SystemExit(f"❌ File not found: {img_path}")
    
    model = YOLO(args.model)
    
    print(f"✓ Model loaded: {args.model}")
    print(f"✓ Confidence threshold: {args.conf}")
    print(f"✓ Deduplication IoU: {args.iou_dedup}")
    print(f"✓ Trash class ID: {args.trash_id}")
    
    test_image(model, img_path, out_dir, args.conf, args.device, 
              args.iou_dedup, args.trash_id)
    
    print(f"\n✓ Results saved to: {out_dir}")
    print(f"  Compare the 3 versions to see the difference!")


if __name__ == "__main__":
    main()
