"""
Sequential training queue with VRAM gate.

Usage:
    python train_queue.py

Add / duplicate entries in RUNS to queue more experiments.
"""
import subprocess
import time

# ── Config ────────────────────────────────────────────────────────────────────
VRAM_CLEAR_MB   = 500   # consider GPU free below this
POLL_INTERVAL_S = 10    # seconds between VRAM checks
GRACE_PERIOD_S  = 15    # wait after run ends before re-checking VRAM

RUNS = [
    [
        "python", "alpha5/train/train_yolo.py",
        "alpha5/datasets/alpha5_trash_v5_restrat/data.yaml", "yolo26x.pt",
        "--epochs",  "500",
        "--batch",   "-1",
        "--imgsz",   "640",
        "--patience","0",
        "--workers", "4",
        "--name",    "YOLO26x_v5_restrat_maxbatch",
    ],
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def vram_used_mb() -> int:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(r.stdout.strip().splitlines()[0])


def wait_vram_clear():
    while True:
        mb = vram_used_mb()
        if mb < VRAM_CLEAR_MB:
            print(f"  [GPU] {mb} MB — clear.")
            return
        print(f"  [GPU] {mb} MB > {VRAM_CLEAR_MB} MB — waiting {POLL_INTERVAL_S}s...")
        time.sleep(POLL_INTERVAL_S)


# ── Queue ─────────────────────────────────────────────────────────────────────
errors = []

for i, cmd in enumerate(RUNS, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(RUNS)}] Waiting for GPU to be free...")
    wait_vram_clear()

    print(f"[{i}/{len(RUNS)}] Starting: {' '.join(cmd[2:])}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = round(time.perf_counter() - t0)

    if result.returncode != 0:
        print(f"  [ERROR] exit code {result.returncode}")
        errors.append((i, result.returncode))
    else:
        print(f"  [OK] finished in {elapsed}s")

    print(f"  Grace period {GRACE_PERIOD_S}s...")
    time.sleep(GRACE_PERIOD_S)

print(f"\n{'='*60}")
if errors:
    print("Finished with errors:")
    for idx, code in errors:
        print(f"  Run {idx}: exit code {code}")
else:
    print(f"All {len(RUNS)} run(s) completed successfully.")
