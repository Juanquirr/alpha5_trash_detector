from ultralytics import YOLO
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from pathlib import Path

MODEL_PATH = "best3.3X.pt"
DATA_YAML = "/app/val_plocan/data.yaml"
SAVE_DIR = "validacion_completa_v3.3"
IMG_SIZE = 640
BATCH_SIZE = 16

model = YOLO(MODEL_PATH)

metrics = model.val(
    data=DATA_YAML,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    conf=0.001,
    iou=0.5,
    save_json=True,
    plots=True,
    save_hybrid=True,
    project=SAVE_DIR,
    name="val_completa"
)

print("=========MÉTRICAS GENERALES=========")
print(f"  mAP@0.50:     {metrics.box.map50:.4f}")
print(f"  mAP@0.50-95:  {metrics.box.map:.4f}")
print(f"  mAP@0.75:     {metrics.box.map75:.4f}")
print(f"  Precision:    {metrics.box.mp:.4f}")
print(f"  Recall:       {metrics.box.mr:.4f}")

# Calcular F1 manualmente
f1_global = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr) if (metrics.box.mp + metrics.box.mr) > 0 else 0
print(f"  F1-Score:     {f1_global:.4f}")

print("=====MÉTRICAS POR CLASE DE BASURA=====")
df_metrics = []

# Métricas por clase
ap5095_per_class = metrics.box.maps  # mAP50-95 por clase

for class_id, class_name in model.names.items():
    map50_95 = ap5095_per_class[class_id] if class_id < len(ap5095_per_class) else 0.0
    
    print(f"  {class_name:10}: mAP50-95={map50_95:0.3f}")
    
    df_metrics.append({
        'Clase': class_name,
        'mAP50-95': float(map50_95)
    })

pd.DataFrame(df_metrics).to_csv(f"{SAVE_DIR}/val_completa/metricas_por_clase.csv", index=False)

print("=====MATRIZ DE CONFUSIÓN=====")
print(f"  Generada automáticamente en: {SAVE_DIR}/val_completa/confusion_matrix.png")

print("=====PREDICCIONES EN VALIDACIÓN=====")
results = model.predict(
    source=f"{DATA_YAML.split('data.yaml')[0]}val/images",
    save_txt=True,
    save_conf=True,
    conf=0.001,
    iou=0.5,
    project=SAVE_DIR,
    name="predicciones_val",
    save=False
)

concat_dir = Path(SAVE_DIR) / "predicciones_val_concat"
concat_dir.mkdir(parents=True, exist_ok=True)

for r in results:
    img_path = r.path
    original_img = cv2.imread(img_path)
    result_img = r.plot()
    
    if original_img is None or result_img is None:
        continue
    
    if original_img.shape[0] != result_img.shape[0]:
        h = min(original_img.shape[0], result_img.shape[0])
        original_img = cv2.resize(original_img, (int(original_img.shape[1] * h / original_img.shape[0]), h))
        result_img = cv2.resize(result_img, (int(result_img.shape[1] * h / result_img.shape[0]), h))
    
    img_out = cv2.hconcat([original_img, result_img])
    out_name = Path(img_path).stem + "_concat.jpg"
    cv2.imwrite(str(concat_dir / out_name), img_out)

print("=====ESTADÍSTICAS COMPLETAS=====")
print(f"  Total imágenes validación: {len(results)}")
print(f"  Precision media: {metrics.box.mp:.4f}")
print(f"  Recall medio: {metrics.box.mr:.4f}")
print(f"  IoU threshold: 0.50")
print(f"  Confidence range: 0.001 - 1.0")
print("==========FINALIZADO==========")
