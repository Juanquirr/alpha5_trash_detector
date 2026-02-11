"""
Alpha5 Visualizer - GUI simplificada para comparar métodos
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import threading
from inference_methods import get_available_methods, get_method

class Alpha5Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Alpha5 Visualizer")
        self.root.geometry("1600x900")

        self.model = None
        self.image = None
        self.results = {}

        self.setup_ui()

    def setup_ui(self):
        # Top: Cargar modelo e imagen
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_frame, text="Modelo:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_label = ttk.Label(top_frame, text="No cargado", foreground="red")
        self.model_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(top_frame, text="Cargar Modelo", command=self.load_model).grid(row=0, column=2, padx=5)

        ttk.Label(top_frame, text="Imagen:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.image_label = ttk.Label(top_frame, text="No cargada", foreground="red")
        self.image_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(top_frame, text="Cargar Imagen", command=self.load_image).grid(row=1, column=2, padx=5)

        # Middle: Selección de métodos y parámetros
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        methods_frame = ttk.LabelFrame(middle_frame, text="Métodos", padding="10")
        methods_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.method_vars = {}
        for i, method_name in enumerate(get_available_methods()):
            method_obj = get_method(method_name)
            var = tk.BooleanVar(value=False)
            self.method_vars[method_name] = var
            ttk.Checkbutton(methods_frame, text=f"{method_obj.name} - {method_obj.description}",
                          variable=var).grid(row=i, column=0, sticky=tk.W, pady=2)

        ttk.Button(methods_frame, text="🚀 Ejecutar", 
                  command=self.run_methods).grid(row=len(get_available_methods()), 
                                                 column=0, pady=10, sticky=tk.EW)

        params_frame = ttk.LabelFrame(middle_frame, text="Parámetros", padding="10")
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        ttk.Label(params_frame, text="Confianza:").grid(row=0, column=0, sticky=tk.W)
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.conf_var,
                 orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(params_frame, textvariable=self.conf_var).grid(row=0, column=2)

        ttk.Label(params_frame, text="IoU:").grid(row=1, column=0, sticky=tk.W)
        self.iou_var = tk.DoubleVar(value=0.45)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.iou_var,
                 orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=tk.EW)
        ttk.Label(params_frame, textvariable=self.iou_var).grid(row=1, column=2)

        params_frame.columnconfigure(1, weight=1)

        # Bottom: Visualización
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Selectores de método
        selector_frame = ttk.Frame(bottom_frame)
        selector_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.method_selectors = []
        for i in range(3):
            frame = ttk.Frame(selector_frame)
            frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            ttk.Label(frame, text=f"Método {i+1}:").pack(side=tk.LEFT)
            selector = ttk.Combobox(frame, state='readonly', width=20)
            selector.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            selector.bind('<<ComboboxSelected>>', lambda e: self.update_display())
            self.method_selectors.append(selector)

        # Canvas para imágenes
        canvas_frame = ttk.Frame(bottom_frame)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvases = []
        self.canvas_labels = []
        for i in range(3):
            frame = ttk.Frame(canvas_frame, relief=tk.RIDGE, borderwidth=2)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

            label = ttk.Label(frame, text=f"Método {i+1}", font=("Arial", 10, "bold"))
            label.pack(side=tk.TOP)
            self.canvas_labels.append(label)

            canvas = tk.Canvas(frame, bg='gray20')
            canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvases.append(canvas)

            stats = ttk.Label(frame, text="", font=("Arial", 8))
            stats.pack(side=tk.BOTTOM)
            self.canvas_labels.append(stats)

        # Status bar
        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, 
                 anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Seleccionar modelo YOLO",
                                             filetypes=[("YOLO weights", "*.pt")])
        if not filepath:
            return
        try:
            self.status_var.set("Cargando modelo...")
            self.root.update()
            self.model = YOLO(filepath)
            self.model_label.config(text=Path(filepath).name, foreground="green")
            self.status_var.set(f"Modelo cargado: {Path(filepath).name}")
            messagebox.showinfo("Éxito", "Modelo cargado correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelo:\n{str(e)}")
            self.status_var.set("Error al cargar modelo")

    def load_image(self):
        filepath = filedialog.askopenfilename(title="Seleccionar imagen",
                                             filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath:
            return
        try:
            self.image = cv2.imread(filepath)
            if self.image is None:
                raise ValueError("No se pudo leer la imagen")
            self.image_label.config(text=Path(filepath).name, foreground="green")
            self.status_var.set(f"Imagen cargada: {Path(filepath).name}")
            self.results = {}
            self.update_method_dropdowns()
            messagebox.showinfo("Éxito", "Imagen cargada correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen:\n{str(e)}")

    def run_methods(self):
        if self.model is None:
            messagebox.showwarning("Advertencia", "Primero carga un modelo")
            return
        if self.image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return

        selected = [name for name, var in self.method_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("Advertencia", "Selecciona al menos un método")
            return

        self.status_var.set("Ejecutando inferencias...")
        self.root.update()
        thread = threading.Thread(target=self._run_thread, args=(selected,))
        thread.start()

    def _run_thread(self, selected_methods):
        params = {'conf': self.conf_var.get(), 'iou': self.iou_var.get(), 'imgsz': 640}

        for method_name in selected_methods:
            try:
                method_obj = get_method(method_name)
                self.status_var.set(f"Ejecutando {method_obj.name}...")
                self.root.update()
                result = method_obj.run(self.image.copy(), self.model, params)
                self.results[method_name] = result
                self.status_var.set(f"{method_obj.name}: {result.num_detections} detecciones ({result.elapsed_time:.2f}s)")
                self.root.update()
            except Exception as e:
                messagebox.showerror("Error", f"Error en {method_name}:\n{str(e)}")

        self.root.after(0, self.update_method_dropdowns)
        self.root.after(0, self.update_display)
        self.status_var.set(f"Completado: {len(selected_methods)} métodos ejecutados")

    def update_method_dropdowns(self):
        available = list(self.results.keys())
        for selector in self.method_selectors:
            selector['values'] = available
            if available:
                selector.current(min(self.method_selectors.index(selector), len(available) - 1))

    def update_display(self):
        for i in range(3):
            selector = self.method_selectors[i]
            if not selector.get():
                continue
            method_name = selector.get()
            if method_name not in self.results:
                continue

            result = self.results[method_name]
            canvas = self.canvases[i]

            img_rgb = cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                img_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img_pil)
            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk, anchor=tk.CENTER)
            canvas.image = img_tk

            method_obj = get_method(method_name)
            self.canvas_labels[i * 2].config(text=f"{method_obj.name}")
            self.canvas_labels[i * 2 + 1].config(
                text=f"Detecciones: {result.num_detections} | Tiempo: {result.elapsed_time:.2f}s"
            )

def main():
    root = tk.Tk()
    app = Alpha5Visualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
