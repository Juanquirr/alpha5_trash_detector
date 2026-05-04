"""
Alpha5 Visualizer - Single-canvas viewer with zoom/pan and result tabs
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from pathlib import Path
from ultralytics import YOLO
import threading
from inference_methods import get_available_methods, get_method


C = {
    'bg':      '#111118',
    'panel':   '#1a1a28',
    'card':    '#252535',
    'accent':  '#e94560',
    'accent2': '#1a6b8a',
    'fg':      '#e8e8f0',
    'dim':     '#666680',
    'border':  '#2a2a40',
}


# ── Zoom/pan canvas ──────────────────────────────────────────────────────────

class ImageViewer:
    """Large canvas with mousewheel zoom-to-cursor and drag pan."""

    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, bg='#0a0a10',
                                highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._bgr = None
        self._photo = None
        self._zoom = 1.0
        self._ox = 0.0   # offset from canvas center
        self._oy = 0.0
        self._drag = None

        self.canvas.bind('<MouseWheel>',       self._wheel)
        self.canvas.bind('<Button-4>',         self._wheel)
        self.canvas.bind('<Button-5>',         self._wheel)
        self.canvas.bind('<ButtonPress-1>',    self._drag_start)
        self.canvas.bind('<B1-Motion>',        self._drag_move)
        self.canvas.bind('<ButtonRelease-1>',  lambda e: setattr(self, '_drag', None))
        self.canvas.bind('<Double-Button-1>',  lambda e: self.fit())
        self.canvas.bind('<Configure>',        lambda e: self._draw())

    def show(self, bgr):
        self._bgr = bgr
        self._zoom = 1.0
        self._ox = self._oy = 0.0
        self.canvas.after(20, self.fit)

    def fit(self):
        if self._bgr is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        ih, iw = self._bgr.shape[:2]
        self._zoom = min(cw / iw, ch / ih)
        self._ox = self._oy = 0.0
        self._draw()

    def _draw(self):
        if self._bgr is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        ih, iw = self._bgr.shape[:2]
        nw = max(1, int(iw * self._zoom))
        nh = max(1, int(ih * self._zoom))

        interp = cv2.INTER_LINEAR if self._zoom > 1 else cv2.INTER_AREA
        resized = cv2.resize(self._bgr, (nw, nh), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas.delete('all')
        self.canvas.create_image(
            cw / 2 + self._ox, ch / 2 + self._oy,
            image=self._photo, anchor=tk.CENTER
        )

    def _wheel(self, e):
        factor = 1.15 if (getattr(e, 'delta', 0) > 0 or getattr(e, 'num', 0) == 4) else 1 / 1.15
        new = self._zoom * factor
        if not 0.04 <= new <= 25.0:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        mx = e.x - cw / 2
        my = e.y - ch / 2
        self._ox = mx + (self._ox - mx) * factor
        self._oy = my + (self._oy - my) * factor
        self._zoom = new
        self._draw()

    def _drag_start(self, e):
        self._drag = (e.x, e.y)

    def _drag_move(self, e):
        if self._drag:
            self._ox += e.x - self._drag[0]
            self._oy += e.y - self._drag[1]
            self._drag = (e.x, e.y)
            self._draw()


# ── Main app ─────────────────────────────────────────────────────────────────

class Alpha5Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title('Alpha5 Visualizer')
        self.root.geometry('1400x820')
        self.root.configure(bg=C['bg'])
        self.root.minsize(900, 600)

        self.model = None
        self.image = None
        self.results = []
        self.selected = -1     # -1 = original

        self._build()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build(self):
        # Sidebar (left)
        sidebar = tk.Frame(self.root, bg=C['panel'], width=270)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Separator
        tk.Frame(self.root, bg=C['border'], width=1).pack(side=tk.LEFT, fill=tk.Y)

        # Right: tabs + viewer + status
        right = tk.Frame(self.root, bg=C['bg'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tab bar
        self.tab_bar = tk.Frame(right, bg=C['panel'], height=44)
        self.tab_bar.pack(side=tk.TOP, fill=tk.X)
        self.tab_bar.pack_propagate(False)

        # Viewer
        self.viewer = ImageViewer(right)

        # Status bar
        self.status = tk.StringVar(value='Load a model and an image to begin.')
        tk.Label(right, textvariable=self.status,
                 bg=C['panel'], fg=C['dim'],
                 font=('Consolas', 9), anchor=tk.W, padx=10, pady=4
                 ).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_sidebar(self, sb):
        def section(text):
            tk.Frame(sb, bg=C['border'], height=1).pack(fill=tk.X, padx=10, pady=(14, 0))
            tk.Label(sb, text=text, bg=C['panel'], fg=C['dim'],
                     font=('Segoe UI', 8, 'bold')).pack(anchor=tk.W, padx=12, pady=(3, 0))

        def btn(text, cmd, accent=False):
            b = tk.Button(sb, text=text, command=cmd,
                          bg=C['accent'] if accent else C['card'],
                          fg=C['fg'], font=('Segoe UI', 10),
                          relief=tk.FLAT, cursor='hand2',
                          activebackground=C['accent2'], activeforeground=C['fg'],
                          padx=8, pady=7)
            b.pack(fill=tk.X, padx=12, pady=3)
            return b

        # Title
        tk.Label(sb, text='Alpha5', font=('Segoe UI', 18, 'bold'),
                 bg=C['panel'], fg=C['accent']).pack(anchor=tk.W, padx=14, pady=(14, 4))
        tk.Label(sb, text='Detection Visualizer', bg=C['panel'], fg=C['dim'],
                 font=('Segoe UI', 9)).pack(anchor=tk.W, padx=14)

        # Model
        section('MODEL')
        self.model_lbl = tk.Label(sb, text='—', bg=C['panel'], fg=C['dim'],
                                   font=('Segoe UI', 8), wraplength=230, anchor=tk.W)
        self.model_lbl.pack(anchor=tk.W, padx=12, pady=2)
        btn('Load Model (.pt)', self.load_model, accent=True)

        # Image
        section('IMAGE')
        self.image_lbl = tk.Label(sb, text='—', bg=C['panel'], fg=C['dim'],
                                   font=('Segoe UI', 8), wraplength=230, anchor=tk.W)
        self.image_lbl.pack(anchor=tk.W, padx=12, pady=2)
        btn('Load Image', self.load_image)

        # Method
        section('METHOD')
        methods = get_available_methods()
        self.method_var = tk.StringVar(value=methods[0] if methods else '')
        cb = ttk.Combobox(sb, textvariable=self.method_var,
                          values=methods, state='readonly', width=28)
        cb.pack(padx=12, pady=6, fill=tk.X)
        cb.bind('<<ComboboxSelected>>', lambda _: self._refresh_params())

        # Params
        section('PARAMETERS')
        self.param_frame = tk.Frame(sb, bg=C['panel'])
        self.param_frame.pack(fill=tk.X, padx=12, pady=4)
        self.param_vars = {}
        self._refresh_params()

        # Run
        tk.Frame(sb, bg=C['panel'], height=6).pack()
        btn('▶  Run Inference', self.run_inference, accent=True)

        # Spacer
        tk.Frame(sb, bg=C['panel']).pack(fill=tk.Y, expand=True)

        # Bottom actions
        tk.Frame(sb, bg=C['border'], height=1).pack(fill=tk.X, padx=10)
        btn('Save Current Image', self.save_current)
        btn('Clear All Results', self.clear_results)

        tk.Label(sb, text='Scroll: zoom  ·  Drag: pan  ·  Dbl-click: fit',
                 bg=C['panel'], fg=C['dim'], font=('Segoe UI', 7)
                 ).pack(pady=(0, 10))

    def _refresh_params(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_vars.clear()

        method = get_method(self.method_var.get())
        if not method:
            return

        SHOW = {'conf', 'iou', 'crops', 'overlap', 'fusion', 'sr_method', 'imgsz'}

        for key, val in method.default_params.items():
            if key not in SHOW:
                continue

            row = tk.Frame(self.param_frame, bg=C['panel'])
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=key, bg=C['panel'], fg=C['dim'],
                     font=('Consolas', 9), width=10, anchor=tk.W).pack(side=tk.LEFT)

            if isinstance(val, bool):
                var = tk.BooleanVar(value=val)
                tk.Checkbutton(row, variable=var, bg=C['panel'],
                               fg=C['fg'], selectcolor=C['card'],
                               activebackground=C['panel']).pack(side=tk.LEFT)
            else:
                var = tk.StringVar(value=str(val))
                tk.Entry(row, textvariable=var, width=10,
                         bg=C['card'], fg=C['fg'],
                         insertbackground=C['fg'], relief=tk.FLAT,
                         font=('Consolas', 9)).pack(side=tk.LEFT, padx=4)

            self.param_vars[key] = (var, type(val))

    def _collect_params(self):
        method = get_method(self.method_var.get())
        params = method.default_params.copy()
        for key, (var, typ) in self.param_vars.items():
            try:
                raw = var.get()
                if typ is bool:
                    params[key] = bool(raw)
                elif typ is int:
                    params[key] = int(float(raw))
                elif typ is float:
                    params[key] = float(raw)
                else:
                    params[key] = raw
            except Exception:
                pass
        return params

    # ── Tab bar ──────────────────────────────────────────────────────────────

    def _rebuild_tabs(self):
        for w in self.tab_bar.winfo_children():
            w.destroy()

        def tab_btn(text, cmd, active):
            bg = C['accent'] if active else C['card']
            fg = C['fg']
            b = tk.Button(self.tab_bar, text=text, command=cmd,
                          bg=bg, fg=fg, font=('Segoe UI', 9),
                          relief=tk.FLAT, cursor='hand2',
                          activebackground=C['accent2'], activeforeground=C['fg'],
                          padx=10, pady=6, bd=0)
            b.pack(side=tk.LEFT, padx=2, pady=5)

        tab_btn('Original', self._show_original, self.selected == -1)

        for i, r in enumerate(self.results):
            label = f'{r.method_name}  ·  {r.num_detections} det  ·  {r.elapsed_time:.1f}s'
            tab_btn(label, lambda i=i: self._show_result(i), self.selected == i)

    # ── Actions ──────────────────────────────────────────────────────────────

    def load_model(self):
        path = filedialog.askopenfilename(
            title='Select YOLO model',
            filetypes=[('YOLO weights', '*.pt'), ('All files', '*.*')]
        )
        if not path:
            return
        try:
            self.status.set('Loading model…')
            self.root.update_idletasks()
            self.model = YOLO(path)
            name = Path(path).name
            self.model_lbl.config(text=name, fg=C['fg'])
            self.status.set(f'Model ready: {name}')
        except Exception as e:
            messagebox.showerror('Model Error', str(e))
            self.status.set('Model load failed.')

    def load_image(self):
        path = filedialog.askopenfilename(
            title='Select image',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp'),
                       ('All files', '*.*')]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror('Image Error', 'Could not read image.')
            return
        self.image = img
        h, w = img.shape[:2]
        name = Path(path).name
        self.image_lbl.config(text=f'{name}  ({w}×{h})', fg=C['fg'])
        self.results.clear()
        self.selected = -1
        self._rebuild_tabs()
        self.viewer.show(img)
        self.status.set(f'Image: {name}  {w}×{h} px')

    def run_inference(self):
        if self.model is None:
            messagebox.showwarning('', 'Load a model first.')
            return
        if self.image is None:
            messagebox.showwarning('', 'Load an image first.')
            return
        method_name = self.method_var.get()
        params = self._collect_params()
        self.status.set(f'Running {method_name}…')
        threading.Thread(
            target=self._infer_thread,
            args=(method_name, params),
            daemon=True
        ).start()

    def _infer_thread(self, method_name, params):
        try:
            result = get_method(method_name).run(self.image.copy(), self.model, params)
            self.root.after(0, self._on_result, result)
        except Exception as e:
            self.root.after(0, lambda: self.status.set(f'Error: {e}'))
            self.root.after(0, lambda: messagebox.showerror('Inference Error', str(e)))

    def _on_result(self, result):
        self.results.append(result)
        self._show_result(len(self.results) - 1)

    def _show_original(self):
        if self.image is None:
            return
        self.selected = -1
        self.viewer.show(self.image)
        self._rebuild_tabs()
        self.status.set('Original image')

    def _show_result(self, idx):
        if not (0 <= idx < len(self.results)):
            return
        self.selected = idx
        r = self.results[idx]
        self.viewer.show(r.image)
        self._rebuild_tabs()
        self.status.set(
            f'{r.method_name}  ·  {r.num_detections} detections  ·  '
            f'{r.elapsed_time:.2f}s  ·  conf={r.params.get("conf", "?")}  ·  '
            f'iou={r.params.get("iou", "?")}'
        )

    def clear_results(self):
        self.results.clear()
        self.selected = -1
        self._rebuild_tabs()
        if self.image is not None:
            self.viewer.show(self.image)
        self.status.set('Results cleared.')

    def save_current(self):
        if self.selected == -1:
            img, name = self.image, 'original.jpg'
        elif 0 <= self.selected < len(self.results):
            img = self.results[self.selected].image
            name = f'{self.results[self.selected].method_name}_result.jpg'
        else:
            messagebox.showwarning('', 'Nothing to save.')
            return
        if img is None:
            messagebox.showwarning('', 'No image loaded.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.jpg',
            filetypes=[('JPEG', '*.jpg'), ('PNG', '*.png'), ('All files', '*.*')],
            initialfile=name
        )
        if path:
            cv2.imwrite(path, img)
            self.status.set(f'Saved: {Path(path).name}')


def main():
    root = tk.Tk()
    Alpha5Visualizer(root)
    root.mainloop()


if __name__ == '__main__':
    main()
