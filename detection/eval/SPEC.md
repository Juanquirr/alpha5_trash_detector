# Specification: common metrics script (`eval_detections.py`)

**Goal.** Measure any detection system (YOLO alone, YOLO + VLM, ...) against the
ground-truth labels with the same rule, so the systems can be compared fairly.
It does not train anything and needs no GPU; it only reads files.

**Inputs.**
- `--gt DIR`: ground-truth labels, one `.txt` per image, YOLO format
  `cls cx cy w h` (normalised).
- `--pred NAME=DIR` (repeatable): predictions of one system, one `.txt` per
  image, `cls cx cy w h conf`. They must be saved with a low confidence
  threshold (conf=0.001), as in Ultralytics validation, or the mAP will come out lower.
- `--images DIR` (optional): the evaluated images. Without it, images that have a
  label file are used; images without a `.txt` count as empty.
- `--names a,b,c` or `--data data.yaml` (optional): class names.
- `--out DIR`: output folder.

**Outputs.** `metrics.json` and `metrics.csv` (one row per system and class, plus
an `all` row), and a table on screen; with several systems, the difference
against the first one.

**Method.** Same as Ultralytics: IoU 0.50 to 0.95 in steps of 0.05; one-to-one
matching by IoU; AP by 101-point interpolation over the precision envelope; P and
R at the confidence that maximises F1 (smoothed curve).

**Acceptance criteria.**
1. Predictions identical to the labels: P = R = 1 and mAP@50 = mAP@50-95 = 0.995
   (the ceiling of the 101-point interpolation).
2. No predictions: everything is 0. A class with no ground truth is left out of the mean.
3. Cases with hand-computed results (a false positive, a miss, a shifted box that
   passes IoU 0.5 but not 0.75) give the expected value.
4. A prediction of another class is not a hit.
5. With several `--pred`, rows have the same format and the difference is shown.
6. Depends only on `numpy` (and `pyyaml` if `--data` is used).
7. Reproduction of the old figure (mAP@50 0.692 ± 0.01) when the user runs it with
   the real YOLO26x predictions on the V10 val set: pending, not verifiable here.

## Verification notes
- The maximum mAP with perfect hits is 0.995 (ceiling of the 101-point
  interpolation, same as Ultralytics); the tests expect this.
- Synthetic tests live in a separate PR (`detection/eval/test_eval_detections.py`), so the logic is verified independently.
- Not yet compared against real Ultralytics (not installed here).
