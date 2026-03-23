# TESTING.md — Testing Practices

## Current State

**Minimal formal testing.** `pytest` is listed as a dependency but no test files (`test_*.py`, `*_test.py`) were found in the codebase.

## Manual Testing (Primary Method)

The `test` subcommand in `run.py` serves as the primary evaluation tool:

```bash
# Compare all inpainting models on 5 random images
python run.py test --model all --max-images 5

# Test a specific model
python run.py test --model redux --num-instances 2

# Test with a specific water detection method
python run.py test --model canny --water-method otsu
```

This generates visual outputs in `outputs_test/{model_name}/` for manual inspection:
- `{stem}_result.png` — generated image
- `{stem}_debug.png` — bounding box visualization
- `{stem}_water_mask.png` — water mask
- `generation_log.csv` — quantitative metadata

## Evaluation Metrics Available

Dependencies include quality metrics (though not wired into the CLI currently):
- `torchmetrics` — standard ML metrics
- `lpips` — Learned Perceptual Image Patch Similarity (perceptual quality)
- `matplotlib` + `seaborn` — visualization for analysis

## Water Detector Comparison

Five water detection methods can be compared manually:
- `hsv` (default) — HSV color space thresholding
- `otsu` — Otsu automatic threshold
- `kmeans` — K-means color clustering
- `flood` — Flood fill from image center
- `sam` — Grounded SAM (most accurate, slowest)

The `water_masks.py` standalone script appears to be a utility for visualizing/comparing water mask outputs.

## Verification Approach

Current quality verification is visual:
1. Run `python run.py test --model all --max-images 5`
2. Inspect outputs in `outputs_test/`
3. Check `_debug.png` overlays for correct object placement
4. Check `_water_mask.png` for water detection quality
5. Compare CSV logs for quantitative metadata

## What Should Be Tested (Gaps)

| Area | Gap |
|------|-----|
| Water detectors | No unit tests for mask quality or edge cases |
| `image_utils.py` | `compute_yolo_bbox`, `compute_crop_region` are pure functions, easy to test |
| `find_water_positions` | Logic is complex (Poisson-like spacing), no tests |
| `run_inpaint` dispatch | Model routing logic untested |
| CLI argument parsing | No smoke tests for subcommands |

## Running Tests (If Added)

```bash
pytest
# or
pytest -v tests/
```

No test configuration file (`pytest.ini`, `pyproject.toml`, `setup.cfg`) currently exists.
