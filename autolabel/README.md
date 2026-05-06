# autolabel — SAM3 Auto-Labelling Pipeline

Automatic YOLO-format label generation using Meta's Segment Anything 3 (SAM3). Runs open-vocabulary trash detection via text prompts and writes `.txt` label files ready for YOLO training.

---

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- NVIDIA GPU with ≥16 GB VRAM (SAM3 is large)
- `transformers>=4.50.0`, `Pillow>=10.0.0`, `numpy>=1.24.0`, `tqdm>=4.65.0`

Install:

```bash
pip install -r autolabel/requirements.txt
```

---

## Docker Setup

```bash
docker build -f autolabel/Dockerfile -t autolabel:latest .
docker run -it --gpus all -v "$(pwd):/app" autolabel:latest
```

Base image: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`.

---

## Download Model

Downloads `facebook/sam3` weights from HuggingFace into the local cache:

```bash
python autolabel/download_model.py
```

Approximately 2.4 GB. Run once before any labelling job.

---

## Usage

### Label a folder of images

```bash
python autolabel/run_autolabel.py images/
```

Labels are written alongside images as `{stem}.txt` (YOLO format). Existing `.txt` files are skipped by default.

### Full YOLO dataset (train/val/test splits)

```bash
python autolabel/run_autolabel.py alpha5_dataset/ --dataset-mode
# Write to separate dir without touching originals:
python autolabel/run_autolabel.py alpha5_dataset/ --dataset-mode --output-root new_dataset/
```

### Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output-labels` | alongside images | Where to write `.txt` files |
| `--output-annot` | — | Save annotated PNG previews for QA |
| `--dataset-mode` | off | Traverse `train/val/test` splits |
| `--output-root` | — | Output root for dataset mode |
| `--classes` | all | Space-separated class IDs to label (e.g. `--classes 0 2`) |
| `--model` | `facebook/sam3` | Local path or HF model ID |
| `--threshold` | 0.3 | Detection confidence threshold |
| `--min-area` | 0.003 | Minimum bbox area as fraction of image area |
| `--max-prompts` | 0 (all) | Limit prompts per class (faster, lower recall) |
| `--no-skip` | off | Re-label images that already have a `.txt` |
| `--list-classes` | — | Print all classes and their prompts, then exit |

### Inspect class definitions

```bash
python autolabel/run_autolabel.py --list-classes
```

---

## Detection Pipeline

For each image and each class:

1. **Text prompts** — multiple per class (defined in `autolabel/prompts.py`), prioritised by `priority` field
2. **SAM3 inference** — `run_prompts_for_class()` returns scored masks
3. **Thresholding** — masks below `--threshold` dropped
4. **Area filter** — bboxes below `--min-area` dropped
5. **NMS** — overlapping boxes merged by IoU (threshold 0.5)
6. **YOLO output** — `cx cy w h` normalised, one line per detection

Outputs per image:
- `{stem}.txt` — YOLO label file
- `{stem}_annot.png` — colour-coded preview (if `--output-annot` set)
- `autolabel_report.json` — per-image counts and scores (written on completion)

---

## Class Definitions

8 classes matching the full pipeline:

| ID | Class | Priority |
|----|-------|----------|
| 0 | `plastic_bottle` | high |
| 1 | `glass` | high |
| 2 | `can` | high |
| 3 | `plastic_bag` | medium |
| 4 | `metal_scrap` | medium |
| 5 | `plastic_wrapper` | medium |
| 6 | `trash_pile` | low |
| 7 | `trash` | low |

Lower-priority classes only run if higher-priority detections don't dominate the scene. `max_prompts_per_class` cuts per-class inference time when recall matters less than speed.

---

## Directory Structure

```
autolabel/
├── Dockerfile
├── requirements.txt
├── run_autolabel.py          CLI entry point
├── download_model.py         HuggingFace model downloader
└── autolabel/                Core package
    ├── __init__.py
    ├── prompts.py            CLASS_DEFS: class IDs, names, prompts, priorities
    ├── model.py              run_prompts_for_class() — SAM3 wrapper
    ├── ops.py                mask_to_bbox(), mask_to_yolo(), nms()
    └── pipeline.py           label_image(), label_folder(), label_dataset()
```
