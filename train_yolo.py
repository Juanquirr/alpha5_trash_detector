import torch
from ultralytics import YOLO
import csv
from datetime import datetime

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"CUDA version en PyTorch: {torch.version.cuda}")
print(f"Número de GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

# Variables para rastrear el mejor mAP50 y la paciencia
best_map50 = 0.0
patience_counter = 0

# Lista para guardar registro de épocas
epoch_log = []

def on_fit_epoch_end(trainer):
    """
    Callback para imprimir y registrar mAP50 de cada época y la paciencia.
    """
    global best_map50, patience_counter
    
    # Obtener mAP50 de las métricas de validación
    current_map50 = trainer.metrics['metrics/mAP50(B)']
    
    # Actualizar best y contador de paciencia
    if current_map50 > best_map50:
        best_map50 = current_map50
        patience_counter = 0
    else:
        patience_counter += 1
    
    print(f"Epoch {trainer.epoch}: mAP50 = {current_map50:.4f}, Best = {best_map50:.4f}, Paciencia = {patience_counter}/15")
    
    # Guardar registro de esta época
    epoch_log.append({
        'epoch': trainer.epoch,
        'map50': current_map50,
        'best_map50': best_map50,
        'patience_counter': patience_counter
    })

# model = YOLO('yolo11n.pt')
# model = YOLO('yolo11m.pt')
model = YOLO('yolo11x.pt')

model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

model.train(
    data="/app/alpha5_trash_v3.3/data.yaml",
    epochs=300,
    batch=-1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    imgsz=640,
    workers=8,
    patience=15,
    ################### HIPERPARAMETROS ####################
    lr0=0.01599,
    lrf=0.00852,
    momentum=0.93365,
    weight_decay=0.00032,
    warmup_epochs=4.87415,
    warmup_momentum=0.95,
    box=5.97732,
    cls=0.57228,
    dfl=1.4791,
    hsv_h=0.0091,
    hsv_s=0.9,
    hsv_v=0.29705,
    degrees=0.0,
    translate=0.08137,
    scale=0.34478,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.25762,
    bgr=0.0,
    mosaic=0.80483,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    close_mosaic=10
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"/runs/detect/train/training_log_{timestamp}_v3.3.csv"

with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['epoch', 'map50', 'best_map50', 'patience_counter']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(epoch_log)

print(f"\n✓ Registro guardado en: {csv_filename}")
print(f"Total de épocas entrenadas: {len(epoch_log)}")
