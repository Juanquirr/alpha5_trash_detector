from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="best3.3X.pt",
    model_device="cuda:0",
    model_confidence_threshold=0.25,
    source="aiplocan",
    slice_height=320,
    slice_width=320,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    project="sahi_batch_outputs"
    )
