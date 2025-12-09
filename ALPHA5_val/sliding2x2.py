from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from ultralytics.utils.plotting import Annotator

model = YOLO("best3.3X.pt")

CONF_THRES = 0.25
IOU_THRES = 0.45
OUTPUTDIR = Path("aiplocan_tiled_inferences")


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def sliding_2x2(img_path):
    img_path_obj = Path(img_path)
    img = cv2.imread(str(img_path_obj))
    h, w = img.shape[:2]

    OUTPUTDIR.mkdir(parents=True, exist_ok=True)
    crops_dir = OUTPUTDIR / f"{img_path_obj.stem}_crops_2x2"
    crops_dir.mkdir(parents=True, exist_ok=True)

    mid_x = w // 2
    mid_y = h // 2

    # Definimos los 4 cuadrantes (x1,y1,x2,y2)
    tiles = [
        (0,      0,      mid_x, mid_y),  # arriba izquierda
        (mid_x,  0,      w,     mid_y),  # arriba derecha
        (0,      mid_y,  mid_x, h),      # abajo izquierda
        (mid_x,  mid_y,  w,     h),      # abajo derecha
    ]

    all_boxes, all_confs, all_classes = [], [], []

    for idx, (x1, y1, x2, y2) in enumerate(tiles):
        patch = img[y1:y2, x1:x2].copy()

        results = model.predict(
            patch,
            verbose=False,
            device="cuda",
            conf=CONF_THRES
        )
        r = results[0]

        # Guardar crop con cajas originales de YOLO
        annotated_patch = r.plot()
        crop_path = crops_dir / f"crop_{idx:02d}.jpg"
        cv2.imwrite(str(crop_path), annotated_patch)

        # Dibujar marco del tile en la imagen completa (debug)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Extraer detecciones y remapear a coords globales
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            for b, c, cls in zip(boxes, confs, clss):
                if c > CONF_THRES:
                    gx1 = b[0] + x1
                    gy1 = b[1] + y1
                    gx2 = b[2] + x1
                    gy2 = b[3] + y1
                    all_boxes.append([gx1, gy1, gx2, gy2])
                    all_confs.append(float(c))
                    all_classes.append(int(cls))

    # NMS global y dibujo con estilo YOLO
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

    out_full = OUTPUTDIR / f"output_{img_path_obj.stem}_2x2.jpg"
    cv2.imwrite(str(out_full), img_annotated)


if __name__ == "__main__":
    sliding_2x2("aiplocan/k10.png")
