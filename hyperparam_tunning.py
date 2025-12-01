import torch
from ultralytics import YOLO

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

model = YOLO('yolo11x.pt')

results = model.tune(
    data="/app/alpha5_trash_v3.3/data.yaml",
    epochs=150,
    iterations=50,
    batch=-1,
    imgsz=640,
    patience=15,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    name="tune_DBv3.3_optimizado",
)

print("\n✓ Búsqueda completada. Mejores hiperparámetros en: runs/detect/tune/best_hyperparameters.yaml")
