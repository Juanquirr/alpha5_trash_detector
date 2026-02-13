"""
Alpha5 Visualizer - Enhanced GUI with zoom and collapsible panels

Enhanced version with:
- Collapsible panels for more image space
- Double-click zoom window with pan and zoom controls
- Interactive image inspection
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


class ZoomWindow:
    """Interactive zoom window for detailed image inspection"""

    def __init__(self, parent, image, method_name, result_info):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Zoom View - {method_name}")
        self.window.geometry("900x700")

        # Store image
        self.original_image = image.copy()
        self.method_name = method_name
        self.result_info = result_info

        # Zoom parameters
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.setup_ui()
        self.display_image()

    def setup_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.window, bg='#2d2d2d', height=50)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Title
        title = tk.Label(toolbar, text=f"{self.method_name}", 
                        font=('Segoe UI', 12, 'bold'),
                        bg='#2d2d2d', fg='#00ff88')
        title.pack(side=tk.LEFT, padx=10)

        # Zoom controls
        controls_frame = tk.Frame(toolbar, bg='#2d2d2d')
        controls_frame.pack(side=tk.RIGHT, padx=10)

        tk.Button(controls_frame, text="🔍+", width=4, 
                 command=lambda: self.adjust_zoom(1.2),
                 bg='#3d3d3d', fg='#e0e0e0', font=('Segoe UI', 10, 'bold'),
                 relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=2)

        tk.Button(controls_frame, text="🔍-", width=4,
                 command=lambda: self.adjust_zoom(0.8),
                 bg='#3d3d3d', fg='#e0e0e0', font=('Segoe UI', 10, 'bold'),
                 relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=2)

        tk.Button(controls_frame, text="↺ Reset", width=8,
                 command=self.reset_view,
                 bg='#3d3d3d', fg='#e0e0e0', font=('Segoe UI', 10, 'bold'),
                 relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=2)

        tk.Button(controls_frame, text="💾 Save", width=8,
                 command=self.save_image,
                 bg='#00aaff', fg='white', font=('Segoe UI', 10, 'bold'),
                 relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=2)

        # Canvas for image
        canvas_frame = tk.Frame(self.window, bg='#1e1e1e')
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg='#3d3d3d', 
                               highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.window.bind("<Escape>", lambda e: self.window.destroy())

        # Status bar
        status_frame = tk.Frame(self.window, bg='#2d2d2d', relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.zoom_label = tk.Label(status_frame, text="Zoom: 100%",
                                   bg='#2d2d2d', fg='#e0e0e0',
                                   font=('Segoe UI', 9), anchor=tk.W, padx=10)
        self.zoom_label.pack(side=tk.LEFT)

        info_label = tk.Label(status_frame, text=self.result_info,
                             bg='#2d2d2d', fg='#00ff88',
                             font=('Segoe UI', 9), anchor=tk.E, padx=10)
        info_label.pack(side=tk.RIGHT)

    def display_image(self):
        """Display image with current zoom and pan"""
        h, w = self.original_image.shape[:2]

        # Calculate zoomed size
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)

        # Resize image
        if self.zoom_level != 1.0:
            resized = cv2.resize(self.original_image, (new_w, new_h), 
                               interpolation=cv2.INTER_LINEAR if self.zoom_level > 1 else cv2.INTER_AREA)
        else:
            resized = self.original_image.copy()

        # Convert to PIL
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1:  # Not yet rendered
            self.window.after(100, self.display_image)
            return

        # Apply pan limits
        max_pan_x = max(0, (new_w - canvas_w) / 2)
        max_pan_y = max(0, (new_h - canvas_h) / 2)
        self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_x))
        self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_y))

        # Create PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)

        # Display on canvas
        self.canvas.delete("all")
        x = canvas_w / 2 + self.pan_x
        y = canvas_h / 2 + self.pan_y
        self.canvas.create_image(x, y, image=self.photo, anchor=tk.CENTER)

        # Update zoom label
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")

    def adjust_zoom(self, factor):
        """Adjust zoom level"""
        new_zoom = self.zoom_level * factor
        if 0.1 <= new_zoom <= 10.0:
            self.zoom_level = new_zoom
            self.display_image()

    def reset_view(self):
        """Reset zoom and pan"""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.display_image()

    def on_mousewheel(self, event):
        """Zoom with mouse wheel"""
        if event.delta > 0:
            self.adjust_zoom(1.1)
        else:
            self.adjust_zoom(0.9)

    def on_drag_start(self, event):
        """Start dragging"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag_motion(self, event):
        """Handle dragging"""
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.display_image()

    def save_image(self):
        """Save current image"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")],
            initialfile=f"{self.method_name}_zoom.jpg"
        )

        if filepath:
            cv2.imwrite(filepath, self.original_image)
            messagebox.showinfo("Success", f"Image saved to:\n{filepath}")


class CollapsibleFrame(ttk.Frame):
    """A frame that can be collapsed/expanded"""

    def __init__(self, parent, text, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)

        self.show = tk.BooleanVar(value=True)

        # Header with toggle button
        self.header_frame = tk.Frame(self, bg='#2d2d2d', relief=tk.RAISED, bd=1)
        self.header_frame.pack(fill=tk.X, padx=2, pady=2)

        self.toggle_button = tk.Label(
            self.header_frame,
            text="▼",
            bg='#2d2d2d',
            fg='#00ff88',
            font=('Segoe UI', 10, 'bold'),
            cursor="hand2",
            width=2
        )
        self.toggle_button.pack(side=tk.LEFT, padx=5)
        self.toggle_button.bind("<Button-1>", self.toggle)

        self.title_label = tk.Label(
            self.header_frame,
            text=text,
            bg='#2d2d2d',
            fg='#00ff88',
            font=('Segoe UI', 11, 'bold'),
            anchor=tk.W
        )
        self.title_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.title_label.bind("<Button-1>", self.toggle)

        # Content frame
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

    def toggle(self, event=None):
        """Toggle frame visibility"""
        if self.show.get():
            self.content_frame.pack_forget()
            self.toggle_button.config(text="▶")
            self.show.set(False)
        else:
            self.content_frame.pack(fill=tk.BOTH, expand=True)
            self.toggle_button.config(text="▼")
            self.show.set(True)


class Alpha5Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Alpha5 Visualizer - Advanced Detection System")
        self.root.geometry("1200x800")

        # Setup dark theme
        self.setup_dark_theme()

        self.model = None
        self.image = None
        self.results = {}
        self.method_params = {}  # Method-specific parameters

        self.setup_ui()

    def setup_dark_theme(self):
        """Setup custom dark theme"""
        style = ttk.Style()
        style.theme_use('clam')

        # Dark theme colors
        bg_dark = '#1e1e1e'
        bg_medium = '#2d2d2d'
        bg_light = '#3d3d3d'
        fg_normal = '#e0e0e0'
        fg_highlight = '#00ff88'
        accent = '#00aaff'

        # Configure styles
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
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_label = ttk.Label(header_frame, text="Alpha5 Visualizer",
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT, pady=(0, 5))

        # Hint label
        hint_label = ttk.Label(header_frame, 
                              text="💡 Tip: Double-click on images to zoom | Collapse panels for more space",
                              foreground=self.colors['fg_highlight'])
        hint_label.pack(side=tk.RIGHT, pady=(0, 5))

        # ============= TOP CONTROLS (Collapsible) =============
        controls_container = ttk.Frame(self.root)
        controls_container.pack(side=tk.TOP, fill=tk.X, padx=10)

        # Model and Image in collapsible frame
        model_image_collapse = CollapsibleFrame(controls_container, "Model & Image")
        model_image_collapse.pack(fill=tk.X, pady=2)

        controls_frame = ttk.Frame(model_image_collapse.content_frame, padding="5")
        controls_frame.pack(fill=tk.X)

        # Model
        model_frame = ttk.LabelFrame(controls_frame, text="Model", padding="10")
        model_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.model_label = ttk.Label(model_frame, text="Not loaded",
                                     foreground=self.colors['accent'])
        self.model_label.pack(side=tk.TOP, pady=5)

        ttk.Button(model_frame, text="Load Model (.pt)",
                  command=self.load_model).pack(side=tk.TOP, fill=tk.X, pady=5)

        # Image
        image_frame = ttk.LabelFrame(controls_frame, text="Image", padding="10")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.image_label = ttk.Label(image_frame, text="Not loaded",
                                     foreground=self.colors['accent'])
        self.image_label.pack(side=tk.TOP, pady=5)

        ttk.Button(image_frame, text="Load Image",
                  command=self.load_image).pack(side=tk.TOP, fill=tk.X, pady=5)

        # ============= MIDDLE: METHODS AND PARAMETERS (Collapsible) =============
        middle_container = ttk.Frame(self.root)
        middle_container.pack(side=tk.TOP, fill=tk.X, padx=10)

        # Methods collapsible
        methods_collapse = CollapsibleFrame(middle_container, "Inference Methods")
        methods_collapse.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        methods_inner = ttk.Frame(methods_collapse.content_frame, padding="5")
        methods_inner.pack(fill=tk.BOTH, expand=True)

        # Scroll for methods
        methods_canvas = tk.Canvas(methods_inner, bg=self.colors['bg_dark'],
                                  highlightthickness=0, height=150)
        methods_scrollbar = ttk.Scrollbar(methods_inner, orient="vertical",
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

            # Frame for each method
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

            # Parameters button
            ttk.Button(method_container, text="⚙️", width=3,
                      command=lambda mn=method_name: self.show_method_params(mn)).pack(side=tk.LEFT, padx=5)

            # Description
            desc_label = ttk.Label(method_container, text=f"- {method_obj.description}",
                                  foreground=self.colors['fg_normal'])
            desc_label.pack(side=tk.LEFT)

            # Initialize default parameters
            self.method_params[method_name] = method_obj.default_params.copy()

        methods_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        methods_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Control buttons
        control_buttons = ttk.Frame(methods_inner)
        control_buttons.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        ttk.Button(control_buttons, text="Run Selected",
                  command=self.run_methods,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ttk.Button(control_buttons, text="Clear Results",
                  command=self.clear_results).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Global parameters collapsible
        params_collapse = CollapsibleFrame(middle_container, "Global Parameters")
        params_collapse.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        params_inner = ttk.Frame(params_collapse.content_frame, padding="10")
        params_inner.pack(fill=tk.BOTH, expand=True)

        # Confidence with entry and slider
        conf_container = ttk.Frame(params_inner)
        conf_container.pack(fill=tk.X, pady=5)

        ttk.Label(conf_container, text="Confidence:").pack(side=tk.LEFT, padx=5)

        self.conf_var = tk.DoubleVar(value=0.25)
        conf_entry = ttk.Entry(conf_container, textvariable=self.conf_var, width=8)
        conf_entry.pack(side=tk.LEFT, padx=5)

        conf_scale = ttk.Scale(conf_container, from_=0.0, to=1.0,
                              variable=self.conf_var, orient=tk.HORIZONTAL,
                              command=self.update_conf_display)
        conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.conf_display = ttk.Label(conf_container, text="0.25")
        self.conf_display.pack(side=tk.LEFT, padx=5)

        # IoU with entry and slider
        iou_container = ttk.Frame(params_inner)
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

        # Apply to all
        ttk.Button(params_inner, text="Apply to All Methods",
                  command=self.apply_global_params).pack(fill=tk.X, pady=10)

        info_label = ttk.Label(params_inner,
                              text="Use ⚙️ for method-specific parameters",
                              foreground=self.colors['fg_highlight'])
        info_label.pack(pady=5)

        # ============= DISPLAY =============
        display_frame = ttk.LabelFrame(self.root, text="Comparative Visualization (Double-click to zoom)", padding="10")
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Method selectors
        selector_frame = ttk.Frame(display_frame)
        selector_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.method_selectors = []
        self.save_buttons = []

        for i in range(3):
            col_frame = ttk.Frame(selector_frame)
            col_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

            ttk.Label(col_frame, text=f"Method {i+1}:").pack(side=tk.TOP, anchor=tk.W)

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

        # Canvas for images
        canvas_container = ttk.Frame(display_frame)
        canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvases = []
        self.canvas_labels = []

        for i in range(3):
            frame = tk.Frame(canvas_container, bg=self.colors['bg_medium'],
                           relief=tk.RAISED, borderwidth=2)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

            label = tk.Label(frame, text=f"Method {i+1}",
                           font=('Segoe UI', 12, 'bold'),
                           bg=self.colors['bg_medium'],
                           fg=self.colors['fg_highlight'])
            label.pack(side=tk.TOP, pady=5)
            self.canvas_labels.append(label)

            canvas = tk.Canvas(frame, bg=self.colors['bg_light'],
                             highlightthickness=0, cursor="hand2")
            canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Bind double-click for zoom
            canvas.bind("<Double-Button-1>", lambda e, idx=i: self.open_zoom_window(idx))

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

        self.status_var = tk.StringVar(value="Ready to use")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                              anchor=tk.W, bg=self.colors['bg_medium'],
                              fg=self.colors['fg_normal'], font=('Segoe UI', 10),
                              padx=10, pady=5)
        status_label.pack(fill=tk.X)

    def open_zoom_window(self, canvas_idx):
        """Open zoom window for detailed inspection"""
        selector = self.method_selectors[canvas_idx]

        if not selector.get():
            messagebox.showinfo("Info", "No image to zoom")
            return

        method_name = selector.get().split(" (")[0]

        if method_name not in self.results:
            messagebox.showwarning("Warning", "Image not found")
            return

        result = self.results[method_name]
        params = self.method_params[method_name]

        # Create info string
        info = f"{result.num_detections} detections | {result.elapsed_time:.2f}s | Conf: {params.get('conf', 0.25):.2f}"

        # Open zoom window
        ZoomWindow(self.root, result.image, method_name, info)

    def update_conf_display(self, value):
        """Update confidence display with 2 decimals"""
        rounded = round(float(value), 2)
        self.conf_var.set(rounded)
        self.conf_display.config(text=f"{rounded:.2f}")

    def update_iou_display(self, value):
        """Update IoU display with 2 decimals"""
        rounded = round(float(value), 2)
        self.iou_var.set(rounded)
        self.iou_display.config(text=f"{rounded:.2f}")

    def show_method_params(self, method_name):
        """Show method-specific parameters window"""
        method_obj = get_method(method_name)

        # Create window
        param_window = tk.Toplevel(self.root)
        param_window.title(f"Parameters: {method_obj.name}")
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

        # Frame for parameters
        params_container = tk.Frame(param_window, bg=self.colors['bg_dark'])
        params_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Temporary variables
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

            if isinstance(param_value, bool):
                var = tk.BooleanVar(value=param_value)
                entry = ttk.Checkbutton(row_frame, variable=var)
                entry.pack(side=tk.LEFT, padx=5)
            elif isinstance(param_value, (int, float)):
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
            """Save modified parameters"""
            new_params = {}
            for param_name, (var, param_type) in param_vars.items():
                try:
                    value = var.get()
                    if param_type == list:
                        import ast
                        new_params[param_name] = ast.literal_eval(value)
                    elif param_type == bool:
                        new_params[param_name] = bool(value)
                    elif param_type == int:
                        new_params[param_name] = int(float(value))
                    elif param_type == float:
                        new_params[param_name] = round(float(value), 2)
                    else:
                        new_params[param_name] = value
                except:
                    new_params[param_name] = value

            self.method_params[method_name] = new_params
            messagebox.showinfo("Success", f"Parameters for {method_obj.name} updated")
            param_window.destroy()

        # Buttons
        button_frame = tk.Frame(param_window, bg=self.colors['bg_dark'])
        button_frame.pack(side=tk.BOTTOM, pady=10)

        ttk.Button(button_frame, text="Save",
                  command=save_params,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Cancel",
                  command=param_window.destroy).pack(side=tk.LEFT, padx=5)

    def apply_global_params(self):
        """Apply global parameters to all methods"""
        conf = round(self.conf_var.get(), 2)
        iou = round(self.iou_var.get(), 2)

        for method_name in self.method_params:
            if 'conf' in self.method_params[method_name]:
                self.method_params[method_name]['conf'] = conf
            if 'iou' in self.method_params[method_name]:
                self.method_params[method_name]['iou'] = iou

        messagebox.showinfo("Applied",
                          f"Global parameters applied:\nConf: {conf:.2f}\nIoU: {iou:.2f}")

    def clear_results(self):
        """Clear results and reset display"""
        self.results = {}

        for canvas in self.canvases:
            canvas.delete("all")

        for selector in self.method_selectors:
            selector.set('')
            selector['values'] = []

        for i, label in enumerate(self.canvas_labels):
            if i % 2 == 0:
                label.config(text=f"Method {i//2 + 1}")
            else:
                label.config(text="")

        self.status_var.set("Results cleared")

    def save_image(self, canvas_idx):
        """Save image from specified canvas"""
        selector = self.method_selectors[canvas_idx]

        if not selector.get():
            messagebox.showwarning("Warning", "No image to save")
            return

        method_name = selector.get().split(" (")[0]

        if method_name not in self.results:
            messagebox.showwarning("Warning", "Image not found")
            return

        result = self.results[method_name]

        filepath = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")],
            initialfile=f"{method_name}_result.jpg"
        )

        if filepath:
            cv2.imwrite(filepath, result.image)
            self.status_var.set(f"Image saved: {Path(filepath).name}")
            messagebox.showinfo("Success", f"Image saved to:\n{filepath}")

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Select YOLO model",
                                             filetypes=[("YOLO weights", "*.pt")])

        if not filepath:
            return

        try:
            self.status_var.set("Loading model...")
            self.root.update()

            self.model = YOLO(filepath)
            self.model_label.config(text=f"{Path(filepath).name}")
            self.status_var.set(f"Model loaded: {Path(filepath).name}")
            messagebox.showinfo("Success", "Model loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading model:\n{str(e)}")
            self.status_var.set("Error loading model")

    def load_image(self):
        filepath = filedialog.askopenfilename(title="Select image",
                                             filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])

        if not filepath:
            return

        try:
            self.image = cv2.imread(filepath)
            if self.image is None:
                raise ValueError("Could not read image")

            self.image_label.config(text=f"{Path(filepath).name}")
            self.status_var.set(f"Image loaded: {Path(filepath).name}")
            self.results = {}
            self.update_method_dropdowns()
            messagebox.showinfo("Success", "Image loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading image:\n{str(e)}")

    def run_methods(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Load a model first")
            return

        if self.image is None:
            messagebox.showwarning("Warning", "Load an image first")
            return

        selected = [name for name, var in self.method_vars.items() if var.get()]

        if not selected:
            messagebox.showwarning("Warning", "Select at least one method")
            return

        self.status_var.set("Running inferences...")
        self.root.update()

        thread = threading.Thread(target=self._run_thread, args=(selected,))
        thread.start()

    def _run_thread(self, selected_methods):
        for method_name in selected_methods:
            try:
                method_obj = get_method(method_name)
                params = self.method_params[method_name].copy()

                self.status_var.set(f"Running {method_obj.name}...")
                self.root.update()

                result = method_obj.run(self.image.copy(), self.model, params)
                self.results[method_name] = result

                self.status_var.set(
                    f"{method_obj.name}: {result.num_detections} detections ({result.elapsed_time:.2f}s)"
                )
                self.root.update()

            except Exception as e:
                messagebox.showerror("Error", f"Error in {method_name}:\n{str(e)}")

        self.root.after(0, self.update_method_dropdowns)
        self.root.after(0, self.update_display)
        self.status_var.set(f"Completed: {len(selected_methods)} methods executed")

    def update_method_dropdowns(self):
        """Update dropdowns with executed methods and their parameters"""
        available = []

        for method_name in self.results.keys():
            method_obj = get_method(method_name)
            params = self.method_params[method_name]

            # Create string with key parameters
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

            # Extract method name (before parenthesis)
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
                    text=f"Detections: {result.num_detections} | "
                         f"Time: {result.elapsed_time:.2f}s | "
                         f"Conf: {params.get('conf', 0.25):.2f}"
                )


def main():
    root = tk.Tk()
    app = Alpha5Visualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
