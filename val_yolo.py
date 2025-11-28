from ultralytics import YOLO
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

MODEL_PATH = "runs/detect/train/weights/best.pt"
DATA_YAML = "/app/alpha5_trash_v3.3/data.yaml"
SAVE_DIR = "validacion_completa_v3.3"
IMG_SIZE = 640
BATCH_SIZE = -1

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
print(f"  mAP@0.50:     {metrics.box.map50:0.4f}")
print(f"  mAP@0.50-95:  {metrics.box.map:0.4f}")
print(f"  mAP@0.75:     {metrics.box.map75:0.4f}")
print(f"  Precision:    {metrics.box.mp:0.4f}")
print(f"  Recall:       {metrics.box.mr:0.4f}")
print(f"  F1-Score:     {metrics.box.f1:0.4f}")

print("=====MÉTRICAS POR CLASE DE BASURA=====")
df_metrics = []
for i, cls_name in enumerate(model.names):
    print(f"  {cls_name:10}: mAP50={metrics.box.maps50[i]:0.3f} | "
          f"P={metrics.box.p[i]:0.3f} | R={metrics.box.r[i]:0.3f} | "
          f"F1={metrics.box.f1[i]:0.3f}")
    df_metrics.append({
        'Clase': cls_name,
        'mAP50': metrics.box.maps50[i],
        'mAP50-95': metrics.box.maps[i],
        'Precision': metrics.box.p[i],
        'Recall': metrics.box.r[i],
        'F1': metrics.box.f1[i]
    })

# Guarda tabla métricas
pd.DataFrame(df_metrics).to_csv(f"{SAVE_DIR}/metricas_por_clase.csv", index=False)

print("=====MATRIZ DE CONFUSIÓN=====")
conf_matrix = model.val(
    data=DATA_YAML,
    conf=0.25,
    iou=0.45,
    save_conf=True,
    plots=True,
    project=SAVE_DIR,
    name="confusion_matrix"
)

print("=====DISTRIBUCIÓN IoU Y CONFIDENCIA=====")
results = model.predict(
    source=f"{DATA_YAML.split('data.yaml')[0]}val/images",
    save_txt=True,
    save_conf=True,
    conf=0.001,
    iou=0.5,
    project=SAVE_DIR,
    name="iou_conf_analysis"
)

print("=====ESTADÍSTICAS COMPLETAS=====")
print(f"  Total imágenes val: {metrics.np}")
print(f"  Predicciones totales: {metrics.box.mp}")
print(f"  IoU threshold usado: 0.50")
print(f"  Confidence range: 0.001 - 1.0")
print("==========FINAL==========")
