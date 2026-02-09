"""
Alpha5 Visualizer - Inference methods comparator
GUI application to visualize and compare small-object detection methods.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import threading

from inference_methods import AVAILABLE_METHODS, get_available_methods, get_method


class Alpha5Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Alpha5 Visualizer - Inference Comparator")
        self.root.geometry("1600x900")

        # State variables
        self.model = None
        self.model_path = None
        self.image = None
        self.image_path = None
        self.video_cap = None
        self.results = {}  # {method_name: InferenceResult}
        self.selected_methods = []

        self.setup_ui()

    def setup_ui(self):
        """Configure UI layout and widgets."""

        # =============== TOP FRAME: Model and image load ===============
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Model
        ttk.Label(top_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_label = ttk.Label(top_frame, text="Not loaded", foreground="red")
        self.model_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(top_frame, text="Load Model (.pt)", 
                  command=self.load_model).grid(row=0, column=2, padx=5)

        # Image / Video
        ttk.Label(top_frame, text="Image:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.image_label = ttk.Label(top_frame, text="Not loaded", foreground="red")
        self.image_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(top_frame, text="Load Image", 
                  command=self.load_image).grid(row=1, column=2, padx=5)
        ttk.Button(top_frame, text="Load Video", 
                  command=self.load_video).grid(row=1, column=3, padx=5)

        # =============== MIDDLE FRAME: Methods selection and global params ===============
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        # Left panel: methods selection
        methods_frame = ttk.LabelFrame(middle_frame, text="Inference Methods", padding="10")
        methods_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.method_vars = {}
        available_methods = get_available_methods()
        for i, method_name in enumerate(available_methods):
            method_obj = get_method(method_name)
            var = tk.BooleanVar(value=False)
            self.method_vars[method_name] = var
            ttk.Checkbutton(
                methods_frame, 
                text=f"{method_obj.name} - {method_obj.description}",
                variable=var
            ).grid(row=i, column=0, sticky=tk.W, pady=2)

        # Run button
        ttk.Button(methods_frame, text="🚀 Run Selected Methods", 
                  command=self.run_selected_methods, 
                  style="Accent.TButton").grid(row=len(available_methods), 
                                               column=0, pady=10, sticky=tk.EW)

        # Right panel: global parameters
        params_frame = ttk.LabelFrame(middle_frame, text="Global Parameters", padding="10")
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        ttk.Label(params_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W)
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.conf_var, 
                 orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(params_frame, textvariable=self.conf_var).grid(row=0, column=2)

        ttk.Label(params_frame, text="IoU:").grid(row=1, column=0, sticky=tk.W)
        self.iou_var = tk.DoubleVar(value=0.45)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.iou_var, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=tk.EW)
        ttk.Label(params_frame, textvariable=self.iou_var).grid(row=1, column=2)

        ttk.Label(params_frame, text="Image Size:").grid(row=2, column=0, sticky=tk.W)
        self.imgsz_var = tk.IntVar(value=640)
        ttk.Combobox(params_frame, textvariable=self.imgsz_var, 
                    values=[320, 480, 640, 960, 1280], 
                    state='readonly').grid(row=2, column=1, sticky=tk.EW)

        params_frame.columnconfigure(1, weight=1)

        # =============== BOTTOM FRAME: Visualization ===============
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # View selector
        view_frame = ttk.Frame(bottom_frame)
        view_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Label(view_frame, text="View:").pack(side=tk.LEFT, padx=5)
        self.view_mode = tk.StringVar(value="single")
        ttk.Radiobutton(view_frame, text="Single", variable=self.view_mode, 
                       value="single", command=self.update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_frame, text="Compare 2", variable=self.view_mode, 
                       value="compare2", command=self.update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_frame, text="Compare 3", variable=self.view_mode, 
                       value="compare3", command=self.update_view).pack(side=tk.LEFT, padx=5)

        # Dropdowns for method selection
        selector_frame = ttk.Frame(bottom_frame)
        selector_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.method_selectors = []
        for i in range(3):
            frame = ttk.Frame(selector_frame)
            frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            ttk.Label(frame, text=f"Method {i+1}:").pack(side=tk.LEFT)
            selector = ttk.Combobox(frame, state='readonly', width=20)
            selector.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            selector.bind('<<ComboboxSelected>>', lambda e, idx=i: self.on_method_selected(idx))
            self.method_selectors.append(selector)

        # Canvases to display images
        canvas_frame = ttk.Frame(bottom_frame)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvases = []
        self.canvas_labels = []
        for i in range(3):
            frame = ttk.Frame(canvas_frame, relief=tk.RIDGE, borderwidth=2)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

            label = ttk.Label(frame, text=f"Method {i+1}", font=("Arial", 10, "bold"))
            label.pack(side=tk.TOP)
            self.canvas_labels.append(label)

            canvas = tk.Canvas(frame, bg='gray20')
            canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvases.append(canvas)

            # Stats label
            stats_label = ttk.Label(frame, text="", font=("Arial", 8))
            stats_label.pack(side=tk.BOTTOM)
            self.canvas_labels.append(stats_label)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_model(self):
        """Load YOLO model from disk."""
        filepath = filedialog.askopenfilename(
            title="Select YOLO model",
            filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            self.status_var.set("Loading model...")
            self.root.update()
            self.model = YOLO(filepath)
            self.model_path = filepath
            self.model_label.config(text=Path(filepath).name, foreground="green")
            self.status_var.set(f"Model loaded: {Path(filepath).name}")
            messagebox.showinfo("Success", "Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model:\n{str(e)}")
            self.status_var.set("Error loading model")

    def load_image(self):
        """Load a single image from disk."""
        filepath = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), 
                      ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            self.image = cv2.imread(filepath)
            if self.image is None:
                raise ValueError("Could not read image from disk.")
            self.image_path = filepath
            self.image_label.config(text=Path(filepath).name, foreground="green")
            self.status_var.set(f"Image loaded: {Path(filepath).name}")

            # Clear previous results
            self.results = {}
            self.update_method_dropdowns()

            messagebox.showinfo("Success", "Image loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image:\n{str(e)}")

    def load_video(self):
        """Load video (placeholder, not yet implemented)."""
        messagebox.showinfo(
            "Info",
            "Video functionality is under development.\n"
            "Frame-by-frame navigation will be implemented here.",
        )

    def run_selected_methods(self):
        """Run all selected inference methods."""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Get selected methods
        selected = [name for name, var in self.method_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("Warning", "Select at least one method.")
            return

        self.status_var.set("Ejecutando inferencias...")
        self.root.update()

        # Ejecutar en thread separado para no bloquear UI
        thread = threading.Thread(target=self._run_inferences_thread, args=(selected,))
        thread.start()

    def _run_inferences_thread(self, selected_methods):
        """Thread worker para ejecutar inferencias"""
        params = {
            'conf': self.conf_var.get(),
            'iou': self.iou_var.get(),
            'imgsz': self.imgsz_var.get()
        }

        for method_name in selected_methods:
            try:
                method_obj = get_method(method_name)
                self.status_var.set(f"Ejecutando {method_obj.name}...")
                self.root.update()

                result = method_obj.run(self.image.copy(), self.model, params)
                self.results[method_name] = result

                self.status_var.set(
                    f"{method_obj.name}: {result.num_detections} detecciones "
                    f"({result.elapsed_time:.2f}s)"
                )
                self.root.update()
            except Exception as e:
                messagebox.showerror("Error", 
                    f"Error en {method_name}:\n{str(e)}")

        # Actualizar dropdowns y vista
        self.root.after(0, self.update_method_dropdowns)
        self.root.after(0, self.update_view)
        self.status_var.set(f"Completed: {len(selected_methods)} methods executed.")

    def update_method_dropdowns(self):
        """Update dropdowns with methods that have results."""
        available = list(self.results.keys())
        for selector in self.method_selectors:
            selector['values'] = available
            if available:
                selector.current(min(self.method_selectors.index(selector), 
                                    len(available) - 1))

    def on_method_selected(self, selector_idx):
        """Callback when a method is selected in one of the dropdowns."""
        self.update_view()

    def update_view(self):
        """Update visualization according to current view mode."""
        mode = self.view_mode.get()

        if mode == "single":
            # Mostrar solo el primer canvas
            for i, canvas in enumerate(self.canvases):
                if i == 0:
                    canvas.master.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
                else:
                    canvas.master.pack_forget()
        elif mode == "compare2":
            # Mostrar 2 canvases
            for i, canvas in enumerate(self.canvases):
                if i < 2:
                    canvas.master.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
                else:
                    canvas.master.pack_forget()
        elif mode == "compare3":
            # Mostrar 3 canvases
            for canvas in self.canvases:
                canvas.master.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Refresh images
        self.display_images()

    def display_images(self):
        """Render images into each visible canvas."""
        mode = self.view_mode.get()
        num_views = {"single": 1, "compare2": 2, "compare3": 3}[mode]

        for i in range(num_views):
            selector = self.method_selectors[i]
            if not selector.get():
                continue

            method_name = selector.get()
            if method_name not in self.results:
                continue

            result = self.results[method_name]
            canvas = self.canvases[i]

            # Convertir imagen para Tkinter
            img_rgb = cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Redimensionar para caber en canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                img_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img_pil)

            # Mostrar en canvas
            canvas.delete("all")
            canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=img_tk, anchor=tk.CENTER
            )
            canvas.image = img_tk  # Mantener referencia

            # Actualizar etiquetas
            method_obj = get_method(method_name)
            self.canvas_labels[i * 2].config(text=f"{method_obj.name}")
            self.canvas_labels[i * 2 + 1].config(
                text=f"Detecciones: {result.num_detections} | "
                     f"Tiempo: {result.elapsed_time:.2f}s"
            )


def main():
    """Punto de entrada principal"""
    root = tk.Tk()
    app = Alpha5Visualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
