# InternImage — Alpha5 Detection Setup

Cascade InternImage-L fine-tuned for Alpha5 trash detection (8 classes).  
This folder contains only **custom files** — configs and dataset class.  
The setup script clones the official InternImage repo and overlays them.

## Classes
```
plastic_bottle, glass, can, plastic_bag,
metal_scrap, plastic_wrapper, trash_pile, trash
```

## Requirements
- Linux / WSL2
- Conda (miniconda or anaconda)
- CUDA 11.8 or 12.1 driver installed
- ~20 GB disk (repo + weights + data)

## 1. Environment setup

```bash
cd InternImage/
bash setup.sh 121    # CUDA 12.1 — change to 118 for CUDA 11.8
```

This will:
1. Create conda env `internimage_alpha5`
2. Install PyTorch 2.1.0 + mmcv-full 1.7.2 + mmdet 2.28.2
3. Clone [OpenGVLab/InternImage](https://github.com/OpenGVLab/InternImage)
4. Compile DCNv3 CUDA ops
5. Copy custom configs and dataset class
6. Create data directory scaffold

## 2. Pretrained weights

Download `internimage_l_22k_192to384.pth` (required before training):

```bash
# Option A: huggingface-cli
huggingface-cli download OpenGVLab/InternImage internimage_l_22k_192to384.pth \
    --local-dir InternImage_repo/detection/pretrained/

# Option B: wget
wget https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22k_192to384.pth \
    -P InternImage_repo/detection/pretrained/
```

Then update `pretrained` path in `configs/coco/cascade_internimage_l_alpha5.py` if needed:
```python
pretrained = './pretrained/internimage_l_22k_192to384.pth'
```

## 3. Dataset

Expected structure inside `InternImage_repo/detection/data/alpha5_coco_v3.3/`:
```
annotations/
    instances_train.json
    instances_val.json
    instances_test.json
images/
    train/
    val/
    test/
```

Copy or symlink your dataset:
```bash
ln -s /path/to/alpha5_coco_v3.3 InternImage_repo/detection/data/alpha5_coco_v3.3
```

## 4. Train

```bash
conda activate internimage_alpha5
cd InternImage_repo/detection

# Single GPU
python tools/train.py configs/coco/cascade_internimage_l_alpha5.py \
    --work-dir work_dirs/cascade_internimage_l_alpha5

# Multi-GPU (e.g. 4 GPUs)
bash tools/dist_train.sh configs/coco/cascade_internimage_l_alpha5.py 4 \
    --work-dir work_dirs/cascade_internimage_l_alpha5
```

## 5. Evaluate

```bash
python tools/test.py configs/coco/cascade_internimage_l_alpha5.py \
    work_dirs/cascade_internimage_l_alpha5/best_bbox_mAP_epoch_*.pth \
    --eval bbox
```

## Notes
- Batch size: 2 per GPU (set in config). Adjust `samples_per_gpu` if OOM.
- `with_cp=True` in backbone — gradient checkpointing active, saves VRAM.
- `SyncBN` in bbox heads — required for multi-GPU; single-GPU works fine too.
- `fp16` enabled — halves VRAM, safe with `loss_scale=dict(init_scale=512)`.
