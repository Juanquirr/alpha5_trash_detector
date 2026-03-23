# Codebase Concerns

**Analysis Date:** 2026-03-23

## Tech Debt

**Massive bloated requirements.txt with zombie dependencies:**
- Issue: `requirements.txt` lists ~30 packages, of which only a small subset is actually imported by any source file. Unused packages include `fastapi`, `uvicorn`, `autodistill`, `autodistill-grounded-sam`, `roboflow`, `openai`, `websocket-client`, `tornado`, `attention_map_diffusers`, `seaborn`, `lpips`, `torchmetrics`, `scikit-learn`, `scipy`, `matplotlib`, `pandas`, `sentencepiece`, `questionary`, `pyfiglet`, `rich`. These appear to be leftovers from earlier experimental phases.
- Files: `requirements.txt`
- Impact: Massively inflated Docker image size (~10+ GB of unnecessary ML libraries), very long build times, potential version conflicts, confusing onboarding.
- Fix approach: Audit each package against actual imports; reduce to the true runtime dependencies: `diffusers`, `transformers`, `torch`, `opencv-python`, `numpy`, `pillow`, `pydantic`, `tqdm`, `accelerate`, `huggingface_hub`, `pytest`.

**`transformers` pinned to bleeding-edge git HEAD:**
- Issue: `requirements.txt` line `transformers @ git+https://github.com/huggingface/transformers.git` installs directly from the default branch, bypassing any version lock.
- Files: `requirements.txt`
- Impact: Non-reproducible builds. Any upstream commit to `huggingface/transformers` can silently break the SAM3 water detector without warning, and there is no way to roll back without knowing the commit hash.
- Fix approach: Pin to a specific release tag or commit SHA: `transformers==4.x.y` or `transformers @ git+...@<sha>`.

**`diffusers` and `accelerate` use open floor version pins:**
- Issue: `diffusers>=0.30.0` and `accelerate>=0.33.0` allow any future major version to be installed, which risks API-breaking upgrades.
- Files: `requirements.txt`
- Impact: Silent breakage when a new major version of `diffusers` changes pipeline APIs (e.g. `FluxFillPipeline` calling convention).
- Fix approach: Pin to a tested range: `diffusers>=0.30.0,<1.0` or pin exact version.

**`FluxCannyInpainter` does not extend the `ImageInpainter` base class:**
- Issue: `FluxFillInpainter` and `FluxReduxInpainter` use `BaseModel` from pydantic alongside `ImageInpainter`. `FluxCannyInpainter` only inherits `BaseModel`, skipping the abstract interface entirely. `FluxKontextInpainter` also skips the base class.
- Files: `core/inpainters/flux_canny.py:22`, `core/inpainters/flux_kontext.py:30`
- Impact: The base class contract in `core/inpainters/base.py` is not enforced for two out of four inpainters. Adding a new required method to the interface would silently fail for these classes at runtime rather than at class definition time.
- Fix approach: Add `ImageInpainter` to the inheritance chain for `FluxCannyInpainter` and `FluxKontextInpainter`.

**`water_masks.py` uses top-level script execution pattern with argparse outside main guard:**
- Issue: `water_masks.py` calls `argparse.parse_args()` and `importlib.import_module()` at module top level — outside any `if __name__ == "__main__":` guard. The `import importlib` statement also appears mid-file after `argparse` parsing (line 35).
- Files: `water_masks.py:27-37`
- Impact: The script cannot be imported as a module without executing side effects (argument parsing, model loading). The mid-file import is a style violation that reduces readability.
- Fix approach: Wrap execution logic inside a `main()` function with a `if __name__ == "__main__":` guard.

**Hard-coded CUDA device strings throughout inpainters:**
- Issue: Every inpainter hard-codes `.to("cuda")` and the SAM detector hard-codes `device: str = "cuda"` as its default.
- Files: `core/inpainters/flux_fill.py:26`, `core/inpainters/flux_canny.py:32`, `core/inpainters/flux_redux.py:57,62`, `core/inpainters/flux_kontext.py:42`, `core/water_detector_sam.py:33`
- Impact: The pipeline is completely non-functional on CPU or MPS (Apple Silicon) environments. There is no graceful fallback or device selection at the entry point. Any debugging or development on a non-GPU machine will fail immediately at model load.
- Fix approach: Accept a `device` parameter propagated from `ProcessConfig` or a top-level CLI argument; default to `"cuda"` but fall back gracefully.

**`run.py` hard-codes `INPUT_DIR = "inputs"` as a relative path:**
- Issue: `cmd_fill()` in `run.py` calls `_collect_images(INPUT_DIR, ...)` where `INPUT_DIR = "inputs"` is a hard-coded relative path string not exposed as a CLI argument.
- Files: `run.py:32`, `run.py:81`
- Impact: The `fill` subcommand silently uses the `inputs/` directory with no override option, while the `test` subcommand has an `--input` flag. This is an inconsistency: users cannot point `fill` at a different input directory.
- Fix approach: Add `--input` argument to the `fill` subcommand consistent with `test`.

**`ProcessConfig` duplicates `min_objects`/`max_objects` constants already in `constants.py`:**
- Issue: `core/constants.py` defines `MIN_OBJECTS = 2` and `MAX_OBJECTS = 3`, but `ProcessConfig` in `core/pipeline.py` re-declares `min_objects: int = 2` and `max_objects: int = 3` as separate defaults.
- Files: `core/pipeline.py:149-150`, `core/constants.py:7-8`
- Impact: If the defaults are changed in one place, the other silently diverges. Currently `constants.py` values `MIN_OBJECTS`/`MAX_OBJECTS` are never imported or used by `pipeline.py`.
- Fix approach: Import `MIN_OBJECTS` and `MAX_OBJECTS` from `core/constants.py` into `ProcessConfig` defaults.

## Known Bugs

**`insert_object` in crop mode can produce incorrect YOLO bbox when mask pixels are empty:**
- Symptoms: If `np.where(mask_np > 127)` yields empty arrays (e.g., a mask where all pixels are below threshold due to Gaussian blur at very small object sizes), the `bbox` assignment on lines 126-131 would raise an `IndexError` on `xs.min()` — but only if the `len(xs) > 0` guard was absent. The guard is present but returns `None`, silently dropping the annotation for that object.
- Files: `core/pipeline.py:125-131`
- Trigger: Very small object sizes combined with aggressive Gaussian blur (`blur_radius=4`) in `create_mask` can push all pixel values below 127, yielding an empty mask.
- Workaround: The object is skipped (no YOLO annotation written), which is silent data loss with no warning logged.

**`process_image` saves outputs even when zero objects are successfully annotated:**
- Symptoms: If all `insert_object` calls return `bbox=None`, `annotations` is empty. The pipeline still writes the synthesized image, an empty `.txt` label file, a debug image, and a water mask to disk.
- Files: `core/pipeline.py:239-244`
- Trigger: Occurs when every object placement fails bbox computation (e.g., edge cases in Kontext diff-detection or empty mask bug above).
- Workaround: None. Results in output images without any YOLO labels, which would corrupt a training dataset if not filtered.

**`FluxKontextInpainter.compute_bbox` depends on pixel-level diff and a fixed threshold:**
- Symptoms: `diff_threshold=25` is baked as a class-level default. For images with subtle water surface changes (reflections, ripples), the diff between the marker image and the result can be noisy, leading to the largest connected component being background change rather than the inserted object.
- Files: `core/inpainters/flux_kontext.py:34`, `core/inpainters/flux_kontext.py:78-113`
- Trigger: Especially unreliable when `use_crop=False` (full-image mode), where any generation variance across the whole image becomes noise in the diff.
- Workaround: Use `use_crop=True` (the default) to minimize diff surface area.

## Security Considerations

**No concerns identified for this offline data generation tool.** The pipeline runs locally with no network-facing endpoints, no user authentication, and no secret credentials in code. Model weights are fetched from Hugging Face Hub with standard authenticated or public access.

## Performance Bottlenecks

**FluxRedux loads two separate full FLUX models into VRAM simultaneously:**
- Problem: `FluxReduxInpainter.model_post_init()` loads both `FluxPriorReduxPipeline` (Redux) and `FluxFillPipeline` (Fill) at init time, keeping both resident in VRAM for the entire run.
- Files: `core/inpainters/flux_redux.py:53-63`
- Cause: Both pipelines are needed per inference call, but loading both at once consumes roughly 2x the VRAM of other inpainters. On the target 32 GB GPU this is feasible but leaves less headroom.
- Improvement path: Offload the Redux prior pipeline to CPU after embedding extraction (`_get_embeddings`) and reload only when needed, or use `enable_model_cpu_offload()` from diffusers.

**K-Means water detector is significantly slower than all other detectors:**
- Problem: `water_detector_kmeans.py` runs `cv2.kmeans` with `attempts=3` on full-resolution feature matrices of shape `(H*W, 4)`. For a 1024px image this is ~1M rows.
- Files: `core/water_detector_kmeans.py:72-76`
- Cause: K-Means is O(n * K * iterations) over pixel count; `attempts=3` runs it 3 times to pick the best result.
- Improvement path: Downsample the image for clustering, then upsample the resulting label map (standard approach). Alternatively reduce `attempts` to 1 for the production `fill` pipeline.

**Each `process_image` call re-runs water detection from scratch:**
- Problem: For the same input image processed with multiple objects, `create_water_mask` is called once per `process_image` invocation. In `cmd_test`, the same image is processed separately for each model (canny, redux, kontext), re-running water detection each time.
- Files: `run.py:142-144`, `core/pipeline.py:183`
- Cause: Water detection result is not cached anywhere between model runs.
- Improvement path: Pre-compute and cache water masks per input image before the model loop in `cmd_test`.

**`find_water_positions` uses a Python loop with `math.hypot` for distance checking:**
- Problem: The minimum-distance check in `core/water_detector.py` iterates over all previously placed positions in a plain Python `for` loop on every candidate point.
- Files: `core/water_detector.py:101-105`
- Cause: Simple O(n * max_attempts) loop; negligible for 2-3 objects but would degrade with more.
- Improvement path: Acceptable for current scale (2-3 objects). Would need a spatial index (KD-tree) for dense placement scenarios.

## Fragile Areas

**SAM3 water detector depends on a non-stable transformers model class:**
- Files: `core/water_detector_sam.py:37-45`
- Why fragile: `Sam3Model` and `Sam3Processor` are imported inside the lazy loader from the `transformers` package pinned to git HEAD. `Sam3` may not yet be released in a stable `transformers` version (as of analysis date), making this detector fragile to any upstream renaming or API change.
- Safe modification: Always test the `sam` water method after any `transformers` upgrade. Do not use as the default method in production.
- Test coverage: No tests exist for any water detector.

**`FluxKontextInpainter` strategy is inherently approximate:**
- Files: `core/inpainters/flux_kontext.py`
- Why fragile: The marker-based positional control (cyan ellipse) is a prompt-engineering trick with no guarantee the model places the object exactly at the marker. The bbox extraction via pixel diff is a heuristic that assumes the largest changed region is the inserted object, which fails if the model also modifies background areas.
- Safe modification: Only use Kontext for qualitative comparison (`test` subcommand), not for the annotated dataset generation (`fill` subcommand). Treat its YOLO bboxes as approximations.
- Test coverage: None.

**`_open_log` in `run.py` opens CSV files in append mode without validation:**
- Files: `run.py:63-69`
- Why fragile: If the output directory already contains a `generation_log.csv` from a previous run with different fields (e.g., from a `fill` run vs a `test` run), appending rows to the existing file will silently mix different schemas without any error.
- Safe modification: Add a schema check when appending to an existing file, or always start a new timestamped log file per run.
- Test coverage: None.

**Crop region computation can produce a degenerate crop smaller than `divisor`:**
- Files: `core/image_utils.py:90-93`
- Why fragile: The edge-case guard at lines 90-93 clamps to `divisor` pixels wide/tall, but does not recompute `crop_x1`/`crop_y1` consistently with `crop_x0`/`crop_y0`, potentially producing a crop of exactly `divisor` pixels regardless of intended size. The downstream `run_inpaint` call then operates on a very small image crop with unpredictable model behavior.
- Safe modification: Add an assertion or early return in `insert_object` when crop dimensions are below a practical minimum (e.g. `MIN_CROP_SIZE`).
- Test coverage: None.

## Scaling Limits

**Dataset generation is single-process and single-GPU:**
- Current capacity: One image processed at a time, sequentially.
- Limit: For large input sets (hundreds of images), total generation time scales linearly. Each image takes ~10-60s per object depending on the model and number of inference steps.
- Scaling path: Parallelize across multiple GPUs by partitioning the input image list and launching separate processes. The pipeline has no shared mutable state that would prevent this.

**Object size constants assume a fixed ~1024px image dimension:**
- Current capacity: `OBJECT_SIZES` in `core/constants.py` specifies pixel sizes (50-200px) calibrated for images resized to `MAX_SIDE=1024`.
- Limit: If `MAX_SIDE` is changed, object sizes become proportionally wrong without any automatic scaling.
- Files: `core/constants.py:12-22`
- Scaling path: Express object sizes as fractions of image dimension rather than absolute pixels.

## Dependencies at Risk

**`transformers` at git HEAD:**
- Risk: Unpinned git dependency breaks reproducibility and can introduce breaking API changes silently.
- Impact: `water_detector_sam.py` becomes non-functional; the `sam` water method breaks entirely.
- Migration plan: Pin to a stable release once `Sam3Model` is included in a `transformers` release.

**`attention_map_diffusers` in requirements.txt (not used):**
- Risk: Package is listed but never imported. It is a small community library with uncertain maintenance status.
- Impact: Increases install fragility with no benefit.
- Migration plan: Remove from `requirements.txt`.

## Missing Critical Features

**No automated tests of any kind:**
- Problem: There are no test files in the project (`pytest` is in requirements but no `test_*.py` files exist anywhere).
- Blocks: Refactoring any water detector, inpainter, or pipeline logic without risk of silent regressions.
- Files affected: All `core/` modules.

**No progress recovery / checkpointing for `fill` runs:**
- Problem: If `cmd_fill` crashes mid-run (e.g., OOM on a large image, or network error on model inference), the entire run must be restarted. The append-mode CSV log partially addresses this for logging, but already-written images are not re-generated (stem conflict would overwrite silently).
- Blocks: Running reliable unattended long dataset generation jobs.

**No validation that `config/prompts.csv` class IDs match `OBJECT_SIZES` keys:**
- Problem: `core/constants.py` defines `OBJECT_SIZES` with keys 0-7, and `config/prompts.csv` has class IDs 0-7. There is no startup check that these are consistent. Adding a new class to the CSV without updating `OBJECT_SIZES` (or vice versa) would cause a `KeyError` at runtime during placement.
- Files: `core/constants.py:12-22`, `core/prompts.py`, `run.py:49-50`

## Test Coverage Gaps

**Water detectors — all five implementations:**
- What's not tested: Correctness of masks on any input image; edge cases (all-sky image, night image, near-zero coverage); consistency of `create_water_mask` signature across all five modules.
- Files: `core/water_detector_hsv.py`, `core/water_detector_otsu.py`, `core/water_detector_kmeans.py`, `core/water_detector_flood.py`, `core/water_detector_sam.py`
- Risk: Silent regression when modifying any detector; broken signature would only be caught at runtime.
- Priority: High

**`find_water_positions` placement logic:**
- What's not tested: Minimum distance enforcement; behavior when water coverage is near the 0.01 threshold; behavior when `n_positions` exceeds available safe pixels.
- Files: `core/water_detector.py:52-126`
- Risk: Object placements could cluster or overlap without detection.
- Priority: High

**`insert_object` / `process_image` pipeline integration:**
- What's not tested: End-to-end correctness of bbox coordinates relative to full-image dimensions; empty mask handling; crop-vs-full-image mode parity.
- Files: `core/pipeline.py`
- Risk: Incorrect YOLO annotations would corrupt the output dataset silently.
- Priority: High

**`image_utils` functions:**
- What's not tested: `compute_crop_region` edge cases (object at image boundary, object larger than `MAX_CROP_SIZE`); `compute_yolo_bbox` with all-zero masks.
- Files: `core/image_utils.py`
- Priority: Medium

---

*Concerns audit: 2026-03-23*
