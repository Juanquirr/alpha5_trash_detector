"""
Usage:
    python run.py --model smolvlm --images images/
    python run.py --model all --images images/          # auto-switches venv per model
    python run.py --model moondream --images images/foto1.jpg
    python run.py --model smolvlm --images images/ --limit 200

Venv reference:
    .transformers-5.X-venv  (transformers 5.x) : smolvlm, qwen_vl, videollama3
    .transformers-4.46-venv (transformers 4.46): moondream, llava, blip2, instructblip,
                                                 clip, paligemma, idefics, mplug_owl3

--model all: spawns a subprocess per model using the correct venv Python automatically.
             Can be run from any Python (system or any venv).
             Resume is per-model — interrupted models restart where they left off.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from models import REGISTRY, VENV
from results import append_row, already_processed, save_prompt_registry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Venv resolution ───────────────────────────────────────────────────────────

def _resolve_python(venv_name: str) -> Path:
    """Find Python executable inside venv_name (relative to this script)."""
    base = Path(__file__).parent / venv_name
    for candidate in [
        base / "Scripts" / "python.exe",   # Windows
        base / "bin" / "python",            # Linux / macOS
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Venv '{venv_name}' not found at {base}.\n"
        f"Run the corresponding setup script first:\n"
        f"  .transformers-5.X-venv  → .\\setup.ps1\n"
        f"  .transformers-4.46-venv → .\\setup_compat.ps1"
    )


# ── Image collection ──────────────────────────────────────────────────────────

def collect_images(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)


# ── Single-model runner (called when --model is a specific key) ───────────────

def run_model(model_key: str, images: list[Path]) -> None:
    csv_path = f"results/detections_{model_key}.csv"

    done    = already_processed(csv_path)
    pending = [img for img in images if img.name not in done]

    if done:
        print(f"\n[{model_key}] Resume: {len(done)} done, {len(pending)} remaining.")
    if not pending:
        print(f"[{model_key}] All images already processed. Nothing to do.")
        return

    cls = REGISTRY[model_key]
    vlm = cls()
    print(f"\n[{model_key}] Loading model...  (venv: {VENV[model_key]})")
    vlm.load()
    print(f"[{model_key}] Processing {len(pending)} image(s)...")

    t_start = time.perf_counter()
    for i, img in enumerate(pending, 1):
        row = vlm.detect_garbage(str(img))
        append_row(row, csv_path=csv_path)

        status  = "YES" if row["garbage_detected"] else "NO "
        classes = row["classes_detected"] or "—"
        elapsed = time.perf_counter() - t_start
        eta     = _fmt_time(elapsed / i * (len(pending) - i))

        print(
            f"  [{i}/{len(pending)}] [{status}] {img.name} | {classes} "
            f"| {row['inference_s']}s | {row['vram_mb']}MB | ETA {eta}"
        )

    vlm.unload()
    save_prompt_registry()
    print(f"[{model_key}] Done in {_fmt_time(time.perf_counter() - t_start)}.")


# ── All-models orchestrator (spawns subprocesses) ─────────────────────────────

def run_all(images_arg: str, limit: int | None) -> None:
    script = Path(__file__).resolve()
    keys   = list(REGISTRY)
    total  = len(keys)
    errors = []

    print(f"\n{'═'*62}")
    print(f"  --model all  →  {total} models queued")
    print(f"  Each model runs in its own venv via subprocess.")
    print(f"  Resume supported: already-processed images are skipped.")
    print(f"{'═'*62}\n")

    for idx, key in enumerate(keys, 1):
        venv = VENV[key]
        print(f"\n[{idx}/{total}] ── {key}  (venv: {venv}) {'─'*(40 - len(key) - len(venv))}")

        try:
            python = _resolve_python(venv)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            errors.append((key, "venv not found"))
            continue

        cmd = [str(python), str(script), "--model", key, "--images", images_arg]
        if limit:
            cmd += ["--limit", str(limit)]

        t0 = time.perf_counter()
        result = subprocess.run(cmd)
        elapsed = _fmt_time(time.perf_counter() - t0)

        if result.returncode != 0:
            print(f"  [ERROR] {key} exited with code {result.returncode} after {elapsed}")
            errors.append((key, f"exit code {result.returncode}"))
        else:
            print(f"  [OK] {key} finished in {elapsed}")

    print(f"\n{'═'*62}")
    if errors:
        print(f"  Finished with errors:")
        for key, reason in errors:
            print(f"    ✗ {key}: {reason}")
    else:
        print(f"  All {total} models completed successfully.")
    print(f"{'═'*62}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True,
                        help=f"Model key, 'all', or comma-separated list. Options: {list(REGISTRY)}")
    parser.add_argument("--images", required=True,
                        help="Image file or directory")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Process only first N images (useful for testing)")
    args = parser.parse_args()

    # ── --model all: orchestrate via subprocesses ──────────────────────────────
    if args.model == "all":
        run_all(args.images, args.limit)
        return

    # ── specific model(s) ──────────────────────────────────────────────────────
    keys = [k.strip() for k in args.model.split(",")]

    images = collect_images(args.images)
    if not images:
        print(f"No images found at: {args.images}")
        sys.exit(1)
    if args.limit:
        images = images[: args.limit]
        print(f"Limit: using first {len(images)} images.")

    for key in keys:
        if key not in REGISTRY:
            print(f"Unknown model '{key}'. Available: {list(REGISTRY)}")
            continue
        run_model(key, images)


if __name__ == "__main__":
    main()
