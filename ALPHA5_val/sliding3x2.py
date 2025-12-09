from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from ultralytics.utils.plotting import Annotator

model = YOLO("best3.3X.pt")

CONF_THRES = 0.25
IOU_THRES = 0.45
OUTPUTDIR = Path("aiplocan_tiled_inferences")

N_COLS = 3  # columnas
N_ROWS = 2  # filas  → 3x2 = 6 tiles


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def sliding_NxM(img_path):
    img_path_obj = Path(img_path)
    img = cv2.imread(str(img_path_obj))
    h, w = img.shape[:2]

    OUTPUTDIR.mkdir(parents=True, exist_ok=True)
    crops_dir = OUTPUTDIR / f"{img_path_obj.stem}_crops_{N_COLS}x{N_ROWS}"
    crops_dir.mkdir(parents=True, exist_ok=True)

    tile_w = w // N_COLS
    tile_h = h // N_ROWS

    all_boxes, all_confs, all_classes = [], [], []

    idx = 0
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x1 = c * tile_w
            y1 = r * tile_h
            x2 = w if c == N_COLS - 1 else (c + 1) * tile_w
            y2 = h if r == N_ROWS - 1 else (r + 1) * tile_h

            patch = img[y1:y2, x1:x2].copy()

            results = model.predict(
                patch,
                verbose=False,
                device="cuda",
                conf=CONF_THRES
            )
            r_res = results[0]

            annotated_patch = r_res.plot()
            crop_path = crops_dir / f"crop_{idx:02d}.jpg"
            cv2.imwrite(str(crop_path), annotated_patch)
            idx += 1

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            if r_res.boxes is not None:
                boxes = r_res.boxes.xyxy.cpu().numpy()
                confs = r_res.boxes.conf.cpu().numpy()
                clss = r_res.boxes.cls.cpu().numpy()
                for b, c_score, cls in zip(boxes, confs, clss):
                    if c_score > CONF_THRES:
                        all_boxes.append([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1])
                        all_confs.append(float(c_score))
                        all_classes.append(int(cls))

    if all_boxes:
        boxes = np.array(all_boxes)
        scores = np.array(all_confs)
        classes = np.array(all_classes)

        keep = []
        for i in range(len(boxes)):
            ok = True
            for j in keep:
                if compute_iou(boxes[i], boxes[j]) > IOU_THRES and scores[i] <= scores[j]:
                    ok = False
                    break
            if ok:
                keep.append(i)

        final_boxes = boxes[keep]
        final_confs = scores[keep]
        final_classes = classes[keep]

        annotator = Annotator(img, line_width=2, example=model.names)
        for box, conf, cls_id in zip(final_boxes, final_confs, final_classes):
            annotator.box_label(box, f"{model.names[cls_id]} {conf:.2f}")
        img_annotated = annotator.result()
    else:
        img_annotated = img

    out_full = OUTPUTDIR / f"output_{img_path_obj.stem}_{N_COLS}x{N_ROWS}.jpg"
    cv2.imwrite(str(out_full), img_annotated)


if __name__ == "__main__":
    sliding_NxM("aiplocan/k10.png")
