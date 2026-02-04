# Alpha5 - Trash Detector

Trash detection system using **YOLO 11**. This repository contains the scripts used for training, validation, and result analysis of a multi-class trash detector trained on a custom labeled dataset.

## Author
- Juan Carlos Rodríguez Ramírez

## Project overview

The goal of this project is to build an **object detection model** robust enough to detect trash across different environments. As a long-term objective, this detector could be integrated into [PLOCAN](https://plocan.eu/) camera streams to automatically detect different types of waste in marine environments.

Key aspects:
- YOLOv11 models trained with **Bayesian hyperparameter tuning**.
- Sliding-window + SAHI inference utilities for handling small objects or complex images.
- Best experiment traces stored under `Yolo11_results/`.


## Environment

The training and validation environment can be reproduced using the provided [Dockerfile](Dockerfile). Build the image and run a container, then execute the training/validation scripts inside the container.

### Build
```bash
docker build -f ./Dockerfile -t yolo11:tag .
```

### Run
```bash
docker run -it --gpus all --shm-size=8g -v "$pwd:/ultralytics/USER" --name "..." yolo11:tag
```

## Dataset

**`ALPHA5_train/alpha5_trash_v3.3/data.yaml`**: Standard Ultralytics YAML defining `train`/`val`/`test` paths and class names.
```
nc: 8
names:
  0: plastic_bottle
  1: glass
  2: can
  3: plastic_bag
  4: metal_scrap
  5: plastic_wrapper
  6: trash_pile
  7: trash
```


## How to use

### 1. Training

#### [train_yolo.py](ALPHA5_train/train_yolo.py)
Main training pipeline with ArgParse, early stopping and mAP50 monitoring per epoch.

```bash
python train_yolo.py data/alpha5_trash_v3.3/data.yaml yolo_model.pt \
  --epochs 300 --batch -1 --imgsz 640 --workers 8 \
  --project /ultralytics/USER/runs/detect/train --name alpha5_yolo11
```

#### [hyperparam_yolo_tunning.py](ALPHA5_train/hyperparam_yolo_tunning.py)
Hyperparameter tuning with `model.tune()`.

```bash
python hyperparam_yolo_tunning.py \
  data/alpha5_trash_v3.3/data.yaml yolo_model.pt 50 20 \
  --imgsz 640 --batch -1 --name alpha5_tune
```

#### [img_stratifier.py](ALPHA5_train/img_stratifier.py)
Instance-stratified dataset split (by object count, not image count).

```bash
python img_stratifier.py mixed_dataset \
  --out_dir alpha5_trash_v3.3/instance_balanced \
  --ratios 0.7 0.2 0.1
```

### 2. Validation & inference

#### [val_yolo.py](ALPHA5_val/val_yolo.py)
Full validation + optional predictions/concat export.

```bash
python val_yolo.py data/alpha5_trash_v3.3/data.yaml runs/detect/train/exp/weights/best.pt \
  --imgsz 640 --batch 16 --conf 0.25 --iou 0.35 \
  --plots --save_json --per_class_csv \
  --predict_val --concat --concat_dirname predictions_val_concat
```

#### [inference.py](ALPHA5_val/inference.py)
Simple inference with time/memory profiling.

```bash
python inference.py images/ yolo_model.pt outputs_inference \
  --device cuda:0 --conf 0.25 --imgsz 640
```

#### [sahi_dir.py](ALPHA5_val/sahi_dir.py)
SAHI sliced inference (size/overlap-driven).

```bash
python sahi_dir.py big_images/ yolo_model.pt sahi_outputs \
  --slice_height 320 --slice_width 320 \
  --overlap_height_ratio 0.2 --overlap_width_ratio 0.2 \
  --device cuda:0 --format jpg --recursive
```

#### [pair_concat.py](ALPHA5_val/pair_concat.py)
Concatenate paired images (original | prediction) by alphabetical order.

```bash
python pair_concat.py originals_dir preds_dir \
  --out_dir concatenated_pairs --suffix _concat
```

#### [crop_maker.py](ALPHA5_val/crop_maker.py)
Generate static crops and/or grid visualization **without inference**.

```bash
python crop_maker.py big_images/ \
  --crops 6 \
  --overlap 0.2 \
  --save_crops \
  --draw_grid \
  --recursive \
  --out_dir image_crops
```

#### [inference_tiled.py](ALPHA5_val/inference_tiled.py)
YOLO inference on uniform crops with NMS/WBF fusion + deduplication + trash class prioritization.

```bash
python inference_tiled.py big_images/ yolo11x.pt \
  --crops 6 \
  --overlap 0.2 \
  --conf 0.25 \
  --iou 0.5 \
  --fusion wbf \
  --iou_dedup 0.5 \
  --trash_id 7 \
  --prioritize_specific \
  --device cuda:0 \
  --save_crops \
  --draw_grid \
  --recursive \
  --out_dir tiled_inferences
```

**Key parameters**:
- `--fusion wbf|nms`: Crop fusion method
- `--iou_dedup`: Final deduplication IoU threshold
- `--trash_id 7 --prioritize_specific`: Prefer specific classes over the id indicated
- `--save_crops`: Save per-crop predictions

#### [patched_inference_alpha.py](ALPHA5_val/patched_inference_alpha.py)
YOLO inference using libraries abstracting logic. Done by Kolesnikov Dmitry. 

```bash
python inference_patched.py source/ model.pt \
    --out_dir patched_results \
    --conf 0.3 \
    --iou 0.5 \
    --patch_size 640 \
    --overlap 0.25 \
    --nms_threshold 0.25 \
    --device cuda:0 \
    --save_comparison \
    --imgsz 640
```

#### [hybrid_pipeline.py](ALPHA5_val/hybrid_pipeline.py)
**Two-stage hybrid pipeline**: Full image → Crops → Smart filtering → WBF merge

```bash
python hybrid_pipeline.py big_images/ yolo11x.pt \
  --crops 6 \
  --overlap 0.2 \
  --conf 0.25 \
  --crops_iou 0.5 \
  --high_iou 0.85 \
  --suspect_iou 0.3 \
  --merge_iou 0.5 \
  --device cuda:0 \
  --save_intermediate \
  --draw_grid \
  --recursive \
  --out_dir hybrid_results
```

**Pipeline stages**:
1. **Full image inference**
2. **Crops inference + WBF**
3. **Smart filtering**: Keep crops detections validated by full (IoU≥0.85) OR new small objects (IoU<0.3)
4. **Final WBF merge**

**Output** (with `--save_intermediate`):
```
*_hybrid_final.jpg          # Final result
*_stage1_full.jpg           # Full image only
*_stage2_crops_raw.jpg      # Crops before filtering
*_stage3_crops_filtered.jpg # Crops after smart filtering
```

## Complete workflow example

```bash
# 1. Generate crops (optional, for inspection)
python crop_maker.py big_images/ --crops 6 --out_dir crops_only

# 2. Run tiled inference (recommended)
python inference_tiled.py big_images/ yolo_model.pt \
  --crops 6 --fusion wbf --out_dir tiled_results

# 3. Run hybrid pipeline (most robust)
python hybrid_pipeline.py big_images/ yolo_model.pt \
  --crops 6 --save_intermediate --out_dir hybrid_results
```

**Recommendation**: Use **`inference_tiled.py`** for most cases (WBF fusion). Use **`hybrid_pipeline.py`** for maximum precision on small objects.

---

## Disclaimer

For this project, **yolo11x.pt** (YOLOv11's heaviest model) was used due to its superior performance on small and complex object detection. Lighter models (`yolo11n/s/m`) were tested but yielded lower mAP50 and recall scores.

Feel free to experiment with:
- Other YOLOv11 sizes: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`
- Ultralytics versions: YOLOv8, YOLOv12  
- Custom models: `from_scratch` or fine-tuning your own checkpoints

**Recommendation**: For limited hardware, start with `yolo11m.pt` and use `--imgsz 416` to balance speed vs accuracy.
