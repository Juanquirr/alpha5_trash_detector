from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="best3.3X.pt",
    confidence_threshold=0.25,
    device="cuda:0",  # o "cpu"
)

result = get_sliced_prediction(
    "aiplocan/k4.png",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Para guardar la imagen con cajas
result.export_visuals(export_dir="sahi_outputs_custom", file_name="k4_sahi")
