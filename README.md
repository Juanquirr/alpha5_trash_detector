# Alpha5 Trash Detector

Trash detection system using **YOLOv11**. This repository contains the scripts used for training, validation, and result analysis of a multi-class trash detector trained on a custom labeled dataset.

---

## Project overview

The goal of this project is to build an **object detection model** robust enough to detect trash across different environments. As a long-term objective, this detector could be integrated into [PLOCAN](https://plocan.eu/) camera streams to automatically detect different types of waste in marine environments.

Key aspects:
- YOLOv11 models trained with **Bayesian hyperparameter tuning**.
- Sliding-window + SAHI inference utilities for handling large images.
- Full experiment traces stored under `Yolo11_results/`.

---

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

- [hyperparam_yolo_tunning.py](ALPHA5_train/hyperparam_yolo_tunning.py)
  This file contains the code to look for hyperparams in a YOLO model, using ArgParse library.
  How to use:
  ```python hyperparam_yolo_tunning.py data.yaml yolo11x.pt 100 20```

- **ALPHA5_train/img_strattifier.py**  
  In this file …

- **ALPHA5_train/alpha5_trash_v3.3/data.yaml**  
  In this file …


### 2. Validation & inference

Validation and evaluation scripts:

- **ALPHA5_val/val_yolo.py**  
  In this file …

- **ALPHA5_val/inference.py**  
  In this file …

- **ALPHA5_val/sahi.py**  
  In this file …

- **ALPHA5_val/sahi_dir.py**
  In this file …
  #### How to control it
  SAHI doesn’t offer a “slice_count=N” parameter; you’d need to compute a slice size/overlap that yields approximately that number for your typical image size (or implement custom slicing and call get_sliced_prediction with your own slice boxes). The built-in API is size/overlap-driven.
  - Fewer crops → increase slice_height/slice_width and/or decrease overlap ratios.​
  - More crops → decrease slice_height/slice_width and/or increase overlap ratios.
  
- [pair_concat.py](ALPHA5_val/pair_concat.py)
  This file..
  How to use:
  ```python pair_concat.py /path/originals /path/preds --match stem --out_dir /path/out```
  ```python pair_concat.py /path/originals /path/preds --match stem --out_dir /path/out```
 