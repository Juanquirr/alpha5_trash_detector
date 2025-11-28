from ultralytics import YOLO
import os
import time
import psutil

# --- Configuración ---
# /app/alpha5_trash_v3/test/images
DIR_ENTRADA = '/app/alpha5_trash_v3_v4/test/images'  # Carpeta con imágenes a analizar
DIR_SALIDA = '/app/salidas_9_DBv3X_fullextras'    # Carpeta donde guardar los resultados
MODELO_PATH = 'runs/detect/train_p15_DBv3X_fullextras/weights/best.pt'

# --- Preparación ---
os.makedirs(DIR_SALIDA, exist_ok=True)
model = YOLO(MODELO_PATH)
process = psutil.Process(os.getpid())

imagenes = [os.path.join(DIR_ENTRADA, f) for f in os.listdir(DIR_ENTRADA)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# --- Procesamiento y Medición ---
for ruta in imagenes:
    # Medir uso de memoria antes de la inferencia
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convertir a MB

    # Medir tiempo de inicio
    start_time = time.monotonic()

    # Inferencia
    results = model(ruta)

    # Medir tiempo de finalización
    end_time = time.monotonic()
    
    # Medir uso de memoria después de la inferencia
    mem_after = process.memory_info().rss / (1024 * 1024)  # Convertir a MB

    # --- Cálculo de métricas ---
    tiempo_procesamiento = end_time - start_time
    incremento_memoria = mem_after - mem_before

    print(f"Imagen: {os.path.basename(ruta)}")
    print(f"  - Tiempo de procesamiento: {tiempo_procesamiento:.4f} segundos")
    print(f"  - Uso de memoria para inferencia: {incremento_memoria:.2f} MB")
    print("-" * 30)

    # Guardar resultados de la detección
    results[0].save(filename=os.path.join(
        DIR_SALIDA, f'detect_{os.path.basename(ruta)}'
    ))

print("Proceso completado.")
