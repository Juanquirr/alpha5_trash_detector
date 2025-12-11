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

best_map50 = 0.0
patience_counter = 0

epoch_log = []

def on_fit_epoch_end(trainer):
    """
    Callback para imprimir y registrar mAP50 de cada época y la paciencia.
    """
    global best_map50, patience_counter
    
    current_map50 = trainer.metrics['metrics/mAP50(B)']
    
    if current_map50 > best_map50:
        best_map50 = current_map50
        patience_counter = 0
    else:
        patience_counter += 1
    
    print(f"Epoch {trainer.epoch}: mAP50 = {current_map50:.4f}, Best = {best_map50:.4f}, Paciencia = {patience_counter}/15")
    
    epoch_log.append({
        'epoch': trainer.epoch,
        'map50': current_map50,
        'best_map50': best_map50,
        'patience_counter': patience_counter
    })

model = YOLO('yolo11x.pt')

model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

model.train(
    data="/ultralytics/plocania/alpha5_trash_v3.3/data.yaml",
    epochs=300,
    batch=-1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    imgsz=640,
    workers=8,
    patience=15,
    project="/ultralytics/plocania/runs/detect/trainPLOCAN", 
)

print(f"Total de épocas entrenadas: {len(epoch_log)}")
