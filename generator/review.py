"""
Quick image reviewer for synthetic dataset selection.

Usage:
    python review.py
    python review.py --input results_PLOCAN_backgrounds --output selected

Keys:
    Y / Space       Accept image
    N / Backspace   Reject image
    Left arrow      Go back one image
    Right arrow     Skip (same as N)
    Q / Escape      Quit and export accepted images
"""

import argparse
import csv
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR  = "results_PLOCAN_backgrounds"
OUTPUT_DIR = "selected"


# ── Collect images ────────────────────────────────────────────────────────────

def collect_debug_images(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    images = sorted(root.glob("**/*_debug.png"))
    return images


def stem_to_synth(debug_path: Path) -> tuple[Path, Path, Path]:
    """Return (synth, txt, debug) paths for a given debug path."""
    stem = debug_path.name.replace("_debug.png", "")
    d = debug_path.parent
    return d / f"{stem}_synth.png", d / f"{stem}.txt", debug_path


# ── Metadata from CSV ─────────────────────────────────────────────────────────

def load_log_index(input_dir: str) -> dict[str, list[str]]:
    """Map synth filename → list of class_name strings."""
    index: dict[str, list[str]] = {}
    for log_path in Path(input_dir).glob("**/generation_log.csv"):
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = Path(row["image_out"]).name
                index.setdefault(key, []).append(row["class_name"])
    return index


# ── Viewer ────────────────────────────────────────────────────────────────────

class Reviewer:
    def __init__(self, images: list[Path], log_index: dict, output_dir: str):
        self.images     = images
        self.log_index  = log_index
        self.output_dir = output_dir
        self.decisions  = [None] * len(images)  # True=accept, False=reject
        self.idx        = 0

        self.root = tk.Tk()
        self.root.title("Image Reviewer")
        self.root.configure(bg="#1a1a1a")
        self.root.attributes("-fullscreen", False)

        # Status bar
        self.status_var = tk.StringVar()
        status = tk.Label(
            self.root, textvariable=self.status_var,
            bg="#1a1a1a", fg="#ffffff",
            font=("Courier", 13), anchor="w", padx=10, pady=6,
        )
        status.pack(side=tk.TOP, fill=tk.X)

        # Class / info bar
        self.info_var = tk.StringVar()
        info = tk.Label(
            self.root, textvariable=self.info_var,
            bg="#1a1a1a", fg="#aaaaaa",
            font=("Courier", 11), anchor="w", padx=10,
        )
        info.pack(side=tk.TOP, fill=tk.X)

        # Image canvas
        self.canvas = tk.Label(self.root, bg="#1a1a1a")
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Legend
        legend = tk.Label(
            self.root,
            text="  Y/Space = Accept    N/Backspace = Reject    ← = Back    Q/Esc = Finish",
            bg="#2a2a2a", fg="#888888",
            font=("Courier", 10), anchor="w", padx=10, pady=4,
        )
        legend.pack(side=tk.BOTTOM, fill=tk.X)

        # Keybindings
        self.root.bind("<y>",          lambda e: self._decide(True))
        self.root.bind("<Y>",          lambda e: self._decide(True))
        self.root.bind("<space>",      lambda e: self._decide(True))
        self.root.bind("<n>",          lambda e: self._decide(False))
        self.root.bind("<N>",          lambda e: self._decide(False))
        self.root.bind("<BackSpace>",  lambda e: self._decide(False))
        self.root.bind("<Right>",      lambda e: self._decide(False))
        self.root.bind("<Left>",       lambda e: self._go_back())
        self.root.bind("<q>",          lambda e: self._finish())
        self.root.bind("<Q>",          lambda e: self._finish())
        self.root.bind("<Escape>",     lambda e: self._finish())
        self.root.bind("<Configure>",  lambda e: self._render())

        self._render()

    def _render(self):
        if self.idx >= len(self.images):
            self._finish()
            return

        path = self.images[self.idx]
        exp  = path.parent.name

        # Load + scale image
        img   = Image.open(path)
        w, h  = img.size
        max_w = max(self.root.winfo_width() - 20, 800)
        max_h = max(self.root.winfo_height() - 120, 500)
        scale = min(max_w / w, max_h / h, 1.0)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=photo)
        self.canvas.image = photo

        # Status
        accepted = sum(1 for d in self.decisions if d is True)
        done     = sum(1 for d in self.decisions if d is not None)
        decision_marker = {True: "✓", False: "✗", None: "·"}[self.decisions[self.idx]]
        self.status_var.set(
            f"  [{self.idx + 1}/{len(self.images)}]  {exp}  {decision_marker}  "
            f"│  accepted: {accepted}  │  remaining: {len(self.images) - done}"
        )

        # Class info from log
        synth_name = path.name.replace("_debug.png", "_synth.png")
        classes    = self.log_index.get(synth_name, [])
        self.info_var.set(f"  classes: {', '.join(classes) or '—'}   │   {path.stem}")

        # Colour title bar by current decision
        colour = {"✓": "#1a3a1a", "✗": "#3a1a1a", "·": "#1a1a1a"}[decision_marker]
        self.root.configure(bg=colour)
        self.canvas.configure(bg=colour)

    def _decide(self, accept: bool):
        self.decisions[self.idx] = accept
        self.idx += 1
        self._render()

    def _go_back(self):
        if self.idx > 0:
            self.idx -= 1
            self._render()

    def _finish(self):
        accepted = [p for p, d in zip(self.images, self.decisions) if d is True]
        if not accepted:
            messagebox.showinfo("Done", "No images accepted.")
            self.root.destroy()
            return

        if not messagebox.askyesno(
            "Export",
            f"Export {len(accepted)} accepted images to '{self.output_dir}/'?"
        ):
            self.root.destroy()
            return

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for debug_path in accepted:
            synth, txt, dbg = stem_to_synth(debug_path)
            exp = debug_path.parent.name
            prefix = exp + "_"
            if synth.exists():
                shutil.copy2(synth, out / (prefix + synth.name))
            if txt.exists():
                shutil.copy2(txt,   out / (prefix + txt.name))
            shutil.copy2(dbg, out / (prefix + dbg.name))

        messagebox.showinfo("Done", f"Exported {len(accepted)} images to {out}/")
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quick synthetic image reviewer")
    parser.add_argument("--input",  default=INPUT_DIR,  help="Results folder")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output folder for accepted images")
    args = parser.parse_args()

    images = collect_debug_images(args.input)
    if not images:
        print(f"No debug images found in {args.input}/")
        return

    print(f"Found {len(images)} images across experiments.")
    log_index = load_log_index(args.input)

    app = Reviewer(images, log_index, args.output)
    app.run()


if __name__ == "__main__":
    main()
