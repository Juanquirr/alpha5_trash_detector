"""
Alpha5 Visualizer - GUI mejorada para comparar métodos
Versión mejorada con controles avanzados y mejor apariencia
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
        self.root.title("Alpha5 Visualizer - Sistema de Detección Avanzado")
        self.root.geometry("1200x800")

        # Configurar tema oscuro
        self.setup_dark_theme()

        self.model = None
        self.image = None
        self.results = {}
        self.method_params = {}  # Parámetros específicos por método

        self.setup_ui()

    def setup_dark_theme(self):
        """Configurar tema oscuro personalizado"""
        style = ttk.Style()
        style.theme_use('clam')

        # Colores del tema oscuro
        bg_dark = '#1e1e1e'
        bg_medium = '#2d2d2d'
        bg_light = '#3d3d3d'
        fg_normal = '#e0e0e0'
        fg_highlight = '#00ff88'
        accent = '#00aaff'

        # Configurar estilos
        style.configure('TFrame', background=bg_dark)
        style.configure('TLabel', background=bg_dark, foreground=fg_normal, font=('Segoe UI', 11))
        style.configure('Title.TLabel', background=bg_dark, foreground=fg_highlight, 
                       font=('Segoe UI', 14, 'bold'))
        style.configure('TButton', background=bg_medium, foreground=fg_normal, 
                       font=('Segoe UI', 10, 'bold'), borderwidth=1)
        style.map('TButton', background=[('active', bg_light)])

        style.configure('Accent.TButton', background=accent, foreground='white',
                       font=('Segoe UI', 11, 'bold'))
        style.map('Accent.TButton', background=[('active', '#0088cc')])

        style.configure('TCheckbutton', background=bg_dark, foreground=fg_normal,
                       font=('Segoe UI', 10))
        style.configure('TLabelframe', background=bg_dark, foreground=fg_highlight,
                       font=('Segoe UI', 11, 'bold'))
        style.configure('TLabelframe.Label', background=bg_dark, foreground=fg_highlight,
                       font=('Segoe UI', 11, 'bold'))

        style.configure('TEntry', fieldbackground=bg_medium, foreground=fg_normal,
                       font=('Segoe UI', 10))
        style.configure('TCombobox', fieldbackground=bg_medium, foreground=fg_normal,
                       font=('Segoe UI', 10))

        self.root.configure(bg=bg_dark)
        self.colors = {
            'bg_dark': bg_dark,
            'bg_medium': bg_medium,
            'bg_light': bg_light,
            'fg_normal': fg_normal,
            'fg_highlight': fg_highlight,
            'accent': accent
        }

    def setup_ui(self):
        # ============= HEADER =============
        header_frame = ttk.Frame(self.root, padding="15")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_label = ttk.Label(header_frame, text="Alpha5 Visualizer", 
                               style='Title.TLabel')
        title_label.pack(side=tk.TOP, pady=(0, 10))

        # ============= TOP CONTROLS =============
        controls_frame = ttk.Frame(self.root, padding="10")
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        # Modelo
        model_frame = ttk.LabelFrame(controls_frame, text="Modelo", padding="10")
        model_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.model_label = ttk.Label(model_frame, text="No cargado", 
                                     foreground=self.colors['accent'])
        self.model_label.pack(side=tk.TOP, pady=5)
        ttk.Button(model_frame, text="Cargar Modelo (.pt)", 
                  command=self.load_model).pack(side=tk.TOP, fill=tk.X, pady=5)

        # Imagen
        image_frame = ttk.LabelFrame(controls_frame, text="Imagen", padding="10")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.image_label = ttk.Label(image_frame, text="No cargada",
                                     foreground=self.colors['accent'])
        self.image_label.pack(side=tk.TOP, pady=5)
        ttk.Button(image_frame, text="Cargar Imagen",
                  command=self.load_image).pack(side=tk.TOP, fill=tk.X, pady=5)

        # ============= MIDDLE: MÉTODOS Y PARÁMETROS =============
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        # Métodos
        methods_frame = ttk.LabelFrame(middle_frame, text="Métodos de Inferencia", padding="10")
        methods_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Scroll para métodos
        methods_canvas = tk.Canvas(methods_frame, bg=self.colors['bg_dark'], 
                                  highlightthickness=0, height=200)
        methods_scrollbar = ttk.Scrollbar(methods_frame, orient="vertical", 
                                         command=methods_canvas.yview)
        methods_scrollable = ttk.Frame(methods_canvas)

        methods_scrollable.bind(
            "<Configure>",
            lambda e: methods_canvas.configure(scrollregion=methods_canvas.bbox("all"))
        )

        methods_canvas.create_window((0, 0), window=methods_scrollable, anchor="nw")
        methods_canvas.configure(yscrollcommand=methods_scrollbar.set)

        self.method_vars = {}
        self.method_param_frames = {}

        for i, method_name in enumerate(get_available_methods()):
            method_obj = get_method(method_name)

            # Frame para cada método
            method_container = ttk.Frame(methods_scrollable)
            method_container.pack(fill=tk.X, pady=3, padx=5)

            # Checkbox
            var = tk.BooleanVar(value=False)
            self.method_vars[method_name] = var

            cb = ttk.Checkbutton(
                method_container,
                text=f"{method_obj.name}",
                variable=var
            )
            cb.pack(side=tk.LEFT)

            # Botón de parámetros
            ttk.Button(method_container, text="⚙️", width=3,
                      command=lambda mn=method_name: self.show_method_params(mn)).pack(side=tk.LEFT, padx=5)

            # Descripción
            desc_label = ttk.Label(method_container, text=f"- {method_obj.description}",
                                  foreground=self.colors['fg_normal'])
            desc_label.pack(side=tk.LEFT)

            # Inicializar parámetros por defecto
            self.method_params[method_name] = method_obj.default_params.copy()

        methods_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        methods_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Botones de control
        control_buttons = ttk.Frame(methods_frame)
        control_buttons.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        ttk.Button(control_buttons, text="Ejecutar Seleccionados",
                  command=self.run_methods,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ttk.Button(control_buttons, text="Limpiar Resultados",
                  command=self.clear_results).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Parámetros globales
        params_frame = ttk.LabelFrame(middle_frame, text="Parámetros Globales", padding="10")
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Confianza con entry y slider
        conf_container = ttk.Frame(params_frame)
        conf_container.pack(fill=tk.X, pady=5)
        ttk.Label(conf_container, text="Confianza:").pack(side=tk.LEFT, padx=5)

        self.conf_var = tk.DoubleVar(value=0.25)
        conf_entry = ttk.Entry(conf_container, textvariable=self.conf_var, width=8)
        conf_entry.pack(side=tk.LEFT, padx=5)

        conf_scale = ttk.Scale(conf_container, from_=0.0, to=1.0, 
                              variable=self.conf_var, orient=tk.HORIZONTAL,
                              command=self.update_conf_display)
        conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.conf_display = ttk.Label(conf_container, text="0.25")
        self.conf_display.pack(side=tk.LEFT, padx=5)

        # IoU con entry y slider
        iou_container = ttk.Frame(params_frame)
        iou_container.pack(fill=tk.X, pady=5)
        ttk.Label(iou_container, text="IoU:").pack(side=tk.LEFT, padx=5)

        self.iou_var = tk.DoubleVar(value=0.45)
        iou_entry = ttk.Entry(iou_container, textvariable=self.iou_var, width=8)
        iou_entry.pack(side=tk.LEFT, padx=5)

        iou_scale = ttk.Scale(iou_container, from_=0.0, to=1.0,
                             variable=self.iou_var, orient=tk.HORIZONTAL,
                             command=self.update_iou_display)
        iou_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.iou_display = ttk.Label(iou_container, text="0.45")
        self.iou_display.pack(side=tk.LEFT, padx=5)

        # Aplicar a todos
        ttk.Button(params_frame, text="Aplicar a Todos los Métodos",
                  command=self.apply_global_params).pack(fill=tk.X, pady=10)

        info_label = ttk.Label(params_frame, 
                              text="Usa ⚙️ para parámetros específicos de cada método",
                              foreground=self.colors['fg_highlight'])
        info_label.pack(pady=5)

        # ============= DISPLAY =============
        display_frame = ttk.LabelFrame(self.root, text="Visualización Comparativa", padding="10")
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Selectores de método
        selector_frame = ttk.Frame(display_frame)
        selector_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.method_selectors = []
        self.save_buttons = []

        for i in range(3):
            col_frame = ttk.Frame(selector_frame)
            col_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

            ttk.Label(col_frame, text=f"Método {i+1}:").pack(side=tk.TOP, anchor=tk.W)

            selector_controls = ttk.Frame(col_frame)
            selector_controls.pack(side=tk.TOP, fill=tk.X)

            selector = ttk.Combobox(selector_controls, state='readonly', width=30)
            selector.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            selector.bind('<<ComboboxSelected>>', lambda e: self.update_display())
            self.method_selectors.append(selector)

            save_btn = ttk.Button(selector_controls, text="💾", width=4,
                                 command=lambda idx=i: self.save_image(idx))
            save_btn.pack(side=tk.LEFT)
            self.save_buttons.append(save_btn)

        # Canvas para imágenes
        canvas_container = ttk.Frame(display_frame)
        canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvases = []
        self.canvas_labels = []

        for i in range(3):
            frame = tk.Frame(canvas_container, bg=self.colors['bg_medium'], 
                           relief=tk.RAISED, borderwidth=2)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

            label = tk.Label(frame, text=f"Método {i+1}", 
                           font=('Segoe UI', 12, 'bold'),
                           bg=self.colors['bg_medium'], 
                           fg=self.colors['fg_highlight'])
            label.pack(side=tk.TOP, pady=5)
            self.canvas_labels.append(label)

            canvas = tk.Canvas(frame, bg=self.colors['bg_light'], 
                             highlightthickness=0)
            canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.canvases.append(canvas)

            stats = tk.Label(frame, text="", font=('Segoe UI', 9),
                           bg=self.colors['bg_medium'],
                           fg=self.colors['fg_normal'])
            stats.pack(side=tk.BOTTOM, pady=5)
            self.canvas_labels.append(stats)

        # ============= STATUS BAR =============
        status_frame = tk.Frame(self.root, bg=self.colors['bg_medium'], 
                               relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="Listo para usar")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               anchor=tk.W, bg=self.colors['bg_medium'],
                               fg=self.colors['fg_normal'], font=('Segoe UI', 10),
                               padx=10, pady=5)
        status_label.pack(fill=tk.X)

    def update_conf_display(self, value):
        """Actualizar display de confianza con 2 decimales"""
        rounded = round(float(value), 2)
        self.conf_var.set(rounded)
        self.conf_display.config(text=f"{rounded:.2f}")

    def update_iou_display(self, value):
        """Actualizar display de IoU con 2 decimales"""
        rounded = round(float(value), 2)
        self.iou_var.set(rounded)
        self.iou_display.config(text=f"{rounded:.2f}")

    def show_method_params(self, method_name):
        """Mostrar ventana de parámetros específicos del método"""
        method_obj = get_method(method_name)

        # Crear ventana
        param_window = tk.Toplevel(self.root)
        param_window.title(f"Parámetros: {method_obj.name}")
        param_window.geometry("500x400")
        param_window.configure(bg=self.colors['bg_dark'])

        # Header
        header = tk.Label(param_window, text=f"⚙️ {method_obj.name}",
                         font=('Segoe UI', 14, 'bold'),
                         bg=self.colors['bg_dark'],
                         fg=self.colors['fg_highlight'])
        header.pack(pady=10)

        desc = tk.Label(param_window, text=method_obj.description,
                       font=('Segoe UI', 10),
                       bg=self.colors['bg_dark'],
                       fg=self.colors['fg_normal'])
        desc.pack(pady=5)

        # Frame para parámetros
        params_container = tk.Frame(param_window, bg=self.colors['bg_dark'])
        params_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Variables temporales
        param_vars = {}

        current_params = self.method_params.get(method_name, method_obj.default_params.copy())

        for i, (param_name, param_value) in enumerate(current_params.items()):
            row_frame = tk.Frame(params_container, bg=self.colors['bg_dark'])
            row_frame.pack(fill=tk.X, pady=5)

            label = tk.Label(row_frame, text=f"{param_name}:",
                           font=('Segoe UI', 10),
                           bg=self.colors['bg_dark'],
                           fg=self.colors['fg_normal'],
                           width=15, anchor=tk.W)
            label.pack(side=tk.LEFT, padx=5)

            if isinstance(param_value, (int, float)):
                var = tk.DoubleVar(value=param_value)
                entry = ttk.Entry(row_frame, textvariable=var, width=10)
                entry.pack(side=tk.LEFT, padx=5)
            elif isinstance(param_value, list):
                var = tk.StringVar(value=str(param_value))
                entry = ttk.Entry(row_frame, textvariable=var, width=30)
                entry.pack(side=tk.LEFT, padx=5)
            elif isinstance(param_value, str):
                var = tk.StringVar(value=param_value)
                entry = ttk.Entry(row_frame, textvariable=var, width=20)
                entry.pack(side=tk.LEFT, padx=5)
            else:
                var = tk.StringVar(value=str(param_value))
                entry = ttk.Entry(row_frame, textvariable=var, width=20)
                entry.pack(side=tk.LEFT, padx=5)

            param_vars[param_name] = (var, type(param_value))

        def save_params():
            """Guardar parámetros modificados"""
            new_params = {}
            for param_name, (var, param_type) in param_vars.items():
                try:
                    value = var.get()
                    if param_type == list:
                        # Convertir string a lista
                        import ast
                        new_params[param_name] = ast.literal_eval(value)
                    elif param_type == int:
                        new_params[param_name] = int(float(value))
                    elif param_type == float:
                        new_params[param_name] = round(float(value), 2)
                    else:
                        new_params[param_name] = value
                except:
                    new_params[param_name] = value

            self.method_params[method_name] = new_params
            messagebox.showinfo("Éxito", f"Parámetros de {method_obj.name} actualizados")
            param_window.destroy()

        # Botones
        button_frame = tk.Frame(param_window, bg=self.colors['bg_dark'])
        button_frame.pack(side=tk.BOTTOM, pady=10)

        ttk.Button(button_frame, text="Guardar", 
                  command=save_params,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar",
                  command=param_window.destroy).pack(side=tk.LEFT, padx=5)

    def apply_global_params(self):
        """Aplicar parámetros globales a todos los métodos"""
        conf = round(self.conf_var.get(), 2)
        iou = round(self.iou_var.get(), 2)

        for method_name in self.method_params:
            if 'conf' in self.method_params[method_name]:
                self.method_params[method_name]['conf'] = conf
            if 'iou' in self.method_params[method_name]:
                self.method_params[method_name]['iou'] = iou

        messagebox.showinfo("Aplicado", 
                          f"Parámetros globales aplicados:\nConf: {conf:.2f}\nIoU: {iou:.2f}")

    def clear_results(self):
        """Limpiar resultados y resetear display"""
        self.results = {}
        for canvas in self.canvases:
            canvas.delete("all")
        for selector in self.method_selectors:
            selector.set('')
            selector['values'] = []
        for i, label in enumerate(self.canvas_labels):
            if i % 2 == 0:
                label.config(text=f"Método {i//2 + 1}")
            else:
                label.config(text="")
        self.status_var.set("Resultados limpiados")

    def save_image(self, canvas_idx):
        """Guardar imagen del canvas especificado"""
        selector = self.method_selectors[canvas_idx]
        if not selector.get():
            messagebox.showwarning("Advertencia", "No hay imagen para guardar")
            return

        method_name = selector.get().split(" (")[0]  # Eliminar parámetros del nombre

        if method_name not in self.results:
            messagebox.showwarning("Advertencia", "Imagen no encontrada")
            return

        result = self.results[method_name]

        filepath = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")],
            initialfile=f"{method_name}_result.jpg"
        )

        if filepath:
            cv2.imwrite(filepath, result.image)
            self.status_var.set(f"Imagen guardada: {Path(filepath).name}")
            messagebox.showinfo("Éxito", f"Imagen guardada en:\n{filepath}")

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Seleccionar modelo YOLO",
                                             filetypes=[("YOLO weights", "*.pt")])
        if not filepath:
            return
        try:
            self.status_var.set("Cargando modelo...")
            self.root.update()
            self.model = YOLO(filepath)
            self.model_label.config(text=f"{Path(filepath).name}")
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
            self.image_label.config(text=f"{Path(filepath).name}")
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
        for method_name in selected_methods:
            try:
                method_obj = get_method(method_name)
                params = self.method_params[method_name].copy()

                self.status_var.set(f"Ejecutando {method_obj.name}...")
                self.root.update()

                result = method_obj.run(self.image.copy(), self.model, params)
                self.results[method_name] = result

                self.status_var.set(
                    f"{method_obj.name}: {result.num_detections} detecciones ({result.elapsed_time:.2f}s)"
                )
                self.root.update()
            except Exception as e:
                messagebox.showerror("Error", f"Error en {method_name}:\n{str(e)}")

        self.root.after(0, self.update_method_dropdowns)
        self.root.after(0, self.update_display)
        self.status_var.set(f"Completado: {len(selected_methods)} métodos ejecutados")

    def update_method_dropdowns(self):
        """Actualizar dropdowns con métodos ejecutados y sus parámetros"""
        available = []
        for method_name in self.results.keys():
            method_obj = get_method(method_name)
            params = self.method_params[method_name]

            # Crear string con parámetros clave
            param_str = f"conf={params.get('conf', 0.25):.2f}"
            if 'iou' in params:
                param_str += f", iou={params.get('iou', 0.45):.2f}"

            display_name = f"{method_name} ({param_str})"
            available.append(display_name)

        for selector in self.method_selectors:
            current = selector.get()
            selector['values'] = available
            if available and not current:
                idx = self.method_selectors.index(selector)
                if idx < len(available):
                    selector.current(idx)

    def update_display(self):
        for i in range(3):
            selector = self.method_selectors[i]
            if not selector.get():
                continue

            # Extraer nombre del método (antes del paréntesis)
            display_name = selector.get()
            method_name = display_name.split(" (")[0]

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
            canvas.create_image(canvas_width // 2, canvas_height // 2, 
                              image=img_tk, anchor=tk.CENTER)
            canvas.image = img_tk

            method_obj = get_method(method_name)
            params = self.method_params[method_name]

            self.canvas_labels[i * 2].config(text=f"{method_obj.name}")
            self.canvas_labels[i * 2 + 1].config(
                text=f"Detecciones: {result.num_detections} | "
                     f"Tiempo: {result.elapsed_time:.2f}s | "
                     f"Conf: {params.get('conf', 0.25):.2f}"
            )

def main():
    root = tk.Tk()
    app = Alpha5Visualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
