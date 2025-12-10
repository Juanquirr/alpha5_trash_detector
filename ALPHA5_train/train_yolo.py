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

# model = YOLO('yolo11n.pt')
# model = YOLO('yolo11m.pt')
model = YOLO('yolo11x.pt')
# model = YOLO('yolo12x.pt')

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
    ################### HIPERPARAMETROS ####################
    # lr0=0.0056,
    # lrf=0.01969,
    # momentum=0.93412,
    # weight_decay=0.0004,
    # warmup_epochs=4.09514,
    # warmup_momentum=0.30372,
    # box=5.69681,
    # cls=0.56072,
    # dfl=2.13634,
    # hsv_h=0.01654,
    # hsv_s=0.85488,
    # hsv_v=0.58432,
    # degrees=0.0,
    # translate=0.08927,
    # scale=0.39442,
    # shear=0.0,
    # perspective=0.0,
    # flipud=0.0,
    # fliplr=0.32289,
    # bgr=0.0,
    # mosaic=0.98711,
    # mixup=0.0,
    # cutmix=0.0,
    # copy_paste=0.0,
    # close_mosaic=10
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"/runs/detect/train/training_log_{timestamp}_v3.3X.csv"

with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['epoch', 'map50', 'best_map50', 'patience_counter']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(epoch_log)

print(f"\n✓ Registro guardado en: {csv_filename}")
print(f"Total de épocas entrenadas: {len(epoch_log)}")