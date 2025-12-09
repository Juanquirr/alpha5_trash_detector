from sahi.predict import get_sliced_prediction
from sahi.auto_model import AutoDetectionModel
from pathlib import Path

# Ruta a tu modelo YOLO11 entrenado
MODEL_PATH = "best3.3X.pt"

# Crear el modelo SAHI + Ultralytics
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",     # integración específica para YOLOv8/11
    model_path=MODEL_PATH,
    confidence_threshold=0.25,
    device="cuda:0",
)

def run_sahi_on_image(image_path, output_dir="sahi_outputs"):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inferencia con slicing (ventanas)
    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=320,
        slice_width=320,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # Guardar visualización con cajas fusionadas por SAHI
    out_name = image_path.stem + "_sahi.png"
    result.export_visuals(export_dir=str(output_dir), file_name=out_name)
    print(f"Guardado: {output_dir / out_name}")

if __name__ == "__main__":
    # Cambia aquí la imagen que quieras probar
    run_sahi_on_image("aiplocan/k3.png")
