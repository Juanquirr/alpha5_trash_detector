"""
pope_run.py  —  Run POPE binary questions through VLM models.

Reads pope_questions/pope_{tier}.jsonl (built by pope_build.py) and writes
pope_results/pope_{model}_{tier}.csv for each (model, tier) combination.

Usage:
    python pope_run.py --model smolvlm --tier random
    python pope_run.py --model smolvlm --tier all
    python pope_run.py --model all --tier all
    python pope_run.py --model smolvlm,llava_ov --tier popular

--model all   spawns one subprocess per model (auto-selects correct venv)
--tier  all   runs random + popular + adversarial

CLIP special case:
    CLIP.describe() ignores the prompt and returns scored lines:
        "plastic bottle: 0.45 | glass: 0.12 | ..."
    pope_run.py parses the queried class score (threshold 0.25) instead of
    calling parse_yesno() on free-text output.

Resume: already-processed question_ids in the output CSV are skipped.

Output CSV columns:
    question_id, image, cls, label, pred, response, inference_s, vram_mb
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from models import REGISTRY, VENV

TIERS      = ("random", "popular", "adversarial")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

POPE_CSV_FIELDS = [
    "question_id", "image", "cls",
    "label", "pred", "response",
    "inference_s", "vram_mb",
]


# ── Yes/no parsers ────────────────────────────────────────────────────────────

def parse_yesno(response: str) -> str:
    """Extract yes/no from first sentence; default yes if no negation found."""
    first = re.split(r"[.!?\n]", response.strip())[0].lower()
    if re.search(r"\bno\b|\bnot\b|\bcannot\b|\bdon't\b|\bdoesn't\b", first):
        return "no"
    return "yes"


def parse_clip_score(response: str, cls: str) -> str:
    """Parse CLIP describe() output (scored candidates) for a specific class."""
    pattern = rf"{re.escape(cls)}:\s*([0-9.]+)"
    match   = re.search(pattern, response, re.IGNORECASE)
    if not match:
        return "no"
    return "yes" if float(match.group(1)) >= 0.25 else "no"


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def _already_done(csv_path: Path) -> set[int]:
    """Return set of question_ids already written to csv_path."""
    if not csv_path.exists():
        return set()
    done: set[int] = set()
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add(int(row["question_id"]))
            except (KeyError, ValueError):
                pass
    return done


def _append_row(row: dict, csv_path: Path) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=POPE_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Question loading ──────────────────────────────────────────────────────────

def load_questions(jsonl_path: Path) -> list[dict]:
    questions = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


# ── Single-model/tier worker ──────────────────────────────────────────────────

def run_model_tier(
    model_key:     str,
    tier:          str,
    questions_dir: Path,
    images_dir:    Path,
    out_dir:       Path,
) -> None:
    import torch

    jsonl_path = questions_dir / f"pope_{tier}.jsonl"
    if not jsonl_path.exists():
        print(f"[{model_key}/{tier}] Question file not found: {jsonl_path}")
        print(f"  Run pope_build.py first.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"pope_{model_key}_{tier}.csv"

    questions = load_questions(jsonl_path)
    done_ids  = _already_done(csv_path)
    pending   = [q for q in questions if q["question_id"] not in done_ids]

    if done_ids:
        print(f"[{model_key}/{tier}] Resume: {len(done_ids)} done, {len(pending)} remaining.")
    if not pending:
        print(f"[{model_key}/{tier}] All questions already processed. Nothing to do.")
        return

    vlm_cls = REGISTRY[model_key]
    vlm     = vlm_cls()
    print(f"\n[{model_key}/{tier}] Loading model...  (venv: {VENV[model_key]})")
    vlm.load()
    print(f"[{model_key}/{tier}] Processing {len(pending)} questions...")

    is_cuda = vlm.device == "cuda" and torch.cuda.is_available()
    is_clip = model_key == "clip"
    t_start = time.perf_counter()

    for i, q in enumerate(pending, 1):
        # Resolve image path (try original name, then alternate extensions)
        image_path = images_dir / q["image"]
        if not image_path.exists():
            found = None
            for ext in IMAGE_EXTS:
                alt = (images_dir / q["image"]).with_suffix(ext)
                if alt.exists():
                    found = alt
                    break
            if not found:
                print(f"  [{i}/{len(pending)}] SKIP (image not found): {q['image']}")
                continue
            image_path = found

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()

        t0       = time.perf_counter()
        response = vlm.describe(str(image_path), q["text"])
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = round(time.perf_counter() - t0, 3)

        vram_mb = 0
        if is_cuda:
            vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

        pred = parse_clip_score(response, q["cls"]) if is_clip else parse_yesno(response)

        _append_row({
            "question_id": q["question_id"],
            "image":       q["image"],
            "cls":         q["cls"],
            "label":       q["label"],
            "pred":        pred,
            "response":    response.strip()[:300],
            "inference_s": elapsed,
            "vram_mb":     vram_mb,
        }, csv_path)

        ok      = "✓" if pred == q["label"] else "✗"
        elapsed_total = time.perf_counter() - t_start
        eta     = _fmt_time(elapsed_total / i * (len(pending) - i))
        print(
            f"  [{i}/{len(pending)}] {ok} q{q['question_id']:05d}  "
            f"{q['image'][:20]:<20s}  {q['cls']:<16s}  "
            f"gt={q['label']}  pred={pred}  |  {elapsed}s  |  ETA {eta}"
        )

    vlm.unload()
    print(f"[{model_key}/{tier}] Done in {_fmt_time(time.perf_counter() - t_start)}.")


# ── Venv resolver ─────────────────────────────────────────────────────────────

def _resolve_python(venv_name: str) -> Path:
    base = Path(__file__).parent / venv_name
    for candidate in [
        base / "Scripts" / "python.exe",
        base / "bin" / "python",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Venv '{venv_name}' not found at {base}.\n"
        f"Run the setup script for this venv first."
    )


# ── All-models orchestrator ───────────────────────────────────────────────────

def run_all(
    model_keys:    list[str],
    tiers:         list[str],
    questions_arg: str,
    images_arg:    str,
    out_arg:       str,
) -> None:
    script  = Path(__file__).resolve()
    combos  = [(k, t) for k in model_keys for t in tiers]
    errors: list[tuple[str, str]] = []

    print(f"\n{'═'*62}")
    print(f"  {len(model_keys)} model(s) × {len(tiers)} tier(s) = {len(combos)} runs")
    print(f"  Models: {', '.join(model_keys)}")
    print(f"  Tiers:  {', '.join(tiers)}")
    print(f"  Each model runs in its own venv via subprocess.")
    print(f"{'═'*62}\n")

    for idx, (key, tier) in enumerate(combos, 1):
        venv = VENV[key]
        print(f"\n[{idx}/{len(combos)}] ── {key}/{tier}  (venv: {venv})")
        try:
            python = _resolve_python(venv)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            errors.append((f"{key}/{tier}", "venv not found"))
            continue

        cmd = [
            str(python), str(script),
            "--model",     key,
            "--tier",      tier,
            "--questions", questions_arg,
            "--images",    images_arg,
            "--out",       out_arg,
        ]
        t0     = time.perf_counter()
        result = subprocess.run(cmd)
        elapsed = _fmt_time(time.perf_counter() - t0)

        if result.returncode != 0:
            print(f"  [ERROR] {key}/{tier} exited with code {result.returncode} after {elapsed}")
            errors.append((f"{key}/{tier}", f"exit code {result.returncode}"))
        else:
            print(f"  [OK] {key}/{tier} finished in {elapsed}")

    print(f"\n{'═'*62}")
    if errors:
        print(f"  Finished with {len(errors)} error(s):")
        for label, reason in errors:
            print(f"    ✗ {label}: {reason}")
    else:
        print(f"  All {len(combos)} runs completed successfully.")
    print(f"{'═'*62}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run POPE binary questions through VLM models")
    parser.add_argument("--model",     required=True,
                        help=f"Model key, 'all', or comma-separated list. Options: {list(REGISTRY)}")
    parser.add_argument("--tier",      default="all",
                        choices=list(TIERS) + ["all"],
                        help="Tier: random / popular / adversarial / all (default: all)")
    parser.add_argument("--questions", default="pope_questions",
                        help="Directory with pope_*.jsonl files (from pope_build.py)")
    parser.add_argument("--images",    default="images",
                        help="Directory containing the image files")
    parser.add_argument("--out",       default="pope_results",
                        help="Output directory for CSV result files")
    args = parser.parse_args()

    tiers = list(TIERS) if args.tier == "all" else [args.tier]

    # ── Multi-model path: spawn subprocesses (one per model for venv isolation) ──
    if args.model == "all":
        run_all(list(REGISTRY), tiers, args.questions, args.images, args.out)
        return

    keys    = [k.strip() for k in args.model.split(",")]
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"Unknown model(s): {unknown}.  Available: {list(REGISTRY)}")
        sys.exit(1)

    if len(keys) > 1:
        run_all(keys, tiers, args.questions, args.images, args.out)
        return

    # ── Single model: run all tiers in current process (correct venv already) ──
    key = keys[0]
    for tier in tiers:
        run_model_tier(
            model_key=key,
            tier=tier,
            questions_dir=Path(args.questions),
            images_dir=Path(args.images),
            out_dir=Path(args.out),
        )


if __name__ == "__main__":
    main()
