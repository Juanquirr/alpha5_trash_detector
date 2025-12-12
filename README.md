# Alpha5 Trash Detector

Trash detection system using **YOLOv11**. This repository contains the scripts used for training, validation, and result analysis of a multi-class trash detector trained on a custom labeled dataset.

---

## Project overview

The goal of this project is to build an **object detection model** robust enough to detect trash across different environments. As a long-term objective, this detector could be integrated into [PLOCAN](https://plocan.eu/) camera streams to automatically detect different types of waste in marine environments.

Key aspects:
- Custom dataset **alpha5_trash_v3.3** with multiple trash classes.
- YOLOv11 models trained with **Bayesian hyperparameter tuning**.
- Sliding-window + SAHI inference utilities for handling large images.
- Full experiment traces stored under `Yolo11_results/`.

---

Your “Environment” section is correct, but it can be made more natural in English and more useful by adding the **exact commands** to build/run the container (so a reviewer can reproduce it immediately). Docker’s own docs emphasize that the Dockerfile is the recipe used to build an image/container, so it’s good to explicitly say “build the image” and “run the container” and then execute scripts inside it.[1][2]

Here’s a polished version you can paste into your README:

## Environment

The training and validation environment can be reproduced using the provided [Dockerfile](Dockerfile). Build the image and run a container, then execute the training/validation scripts inside the container.

### Build
```
docker build -f .\Dockerfile -t yolo11:tag .
```

### Run
```
docker run -it --gpus all --shm-size=8g -v "$pwd:/ultralytics/USER" --name "..." yolo11:tag
```

Inside the container:
```
python train_args_yolo.py
python val_args_yolo.py
```

---

## How to use

### 1. Training

Main training pipeline:

- [train_args_yolo.py](ALPHA5_train/train_args_yolo.py)
  This file contains the main code for training a model, using ArgParse library.
  How to use:
  ```python train_args_yolo.py data.yaml yolo11x.pt```
  
- [train_yolo.py](ALPHA5_train/train_yolo.py)
  This file contains the same code than [train_args_yolo.py](ALPHA5_train/train_args_yolo.py), but without ArgParse library.
  How to use:
  ```python train_yolo.py```

- **ALPHA5_train/hyperparam_tunning.py**  
  In this file …

- **ALPHA5_train/img_strattifier.py**  
  In this file …

- **ALPHA5_train/alpha5_trash_v3.3/data.yaml**  
  In this file …


### 2. Validation & inference

Validation and evaluation scripts:

- `ALPHA5_val/val_yolo.py` – validation with metrics (mAP, IoU, confusion matrix, etc.).
- `ALPHA5_val/inference.py` – inference on single images or folders.
- `ALPHA5_val/sahi.py`, `sahi_dir.py`, `sahi_test.py` – SAHI-based inference.
- `ALPHA5_val/sliding2x2.py`, `sliding3x2.py` – sliding-window tiling strategies.

---

## File descriptions

Use this format to document each script/config in more detail:



- **ALPHA5_val/val_yolo.py**  
  In this file …

- **ALPHA5_val/inference.py**  
  In this file …

- **ALPHA5_val/sahi.py**  
  In this file …

- **ALPHA5_val/sahi_dir.py**  
  In this file …

- **AL
 