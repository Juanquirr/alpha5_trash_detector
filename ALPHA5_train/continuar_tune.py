from ultralytics import YOLO

model = YOLO('yolo11x.pt')

# 2. Reanuda el hyperparameter tuning en el mismo directorio
model.tune(
    data="/app/alpha5_trash_v3.3/data.yaml",
    name="tune_DBv3.3_optimizado",   # EXACTAMENTE el nombre de esa carpeta
    resume=True,                     # clave para continuar la sesión
    exist_ok=True,                   # permite reutilizar el directorio
    epochs=150,
    iterations=60,
    batch=16,
    imgsz=640,
    patience=15,
    device=0,
)
