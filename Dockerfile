FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    "transformers>=4.50.0" \
    "huggingface_hub>=0.20.0" \
    "Pillow>=10.0.0" \
    "numpy>=1.24.0" \
    "tqdm>=4.65.0"

COPY . .
