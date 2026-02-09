"""
Script de ejemplo para probar métodos sin GUI
"""
import cv2
import numpy as np
from ultralytics import YOLO
from ALPHA5_val.inference_methods import get_method, get_available_methods


def test_methods(image_path, model_path, output_dir="test_results"):
    """
    Ejecuta todos los métodos y guarda resultados
    """
    from pathlib import Path

    # Cargar modelo e imagen
    print(f"Cargando modelo: {model_path}")
    model = YOLO(model_path)

    print(f"Cargando imagen: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar imagen: {image_path}")

    # Crear directorio de salida
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    # Parámetros globales
    params = {
        'conf': 0.25,
        'iou': 0.45,
        'imgsz': 640
    }

    print(f"\n{'='*60}")
    print("EJECUTANDO MÉTODOS DE INFERENCIA")
    print(f"{'='*60}\n")

    results = {}
    for method_name in get_available_methods():
        try:
            method_obj = get_method(method_name)
            print(f"Ejecutando {method_obj.name}...")

            result = method_obj.run(image.copy(), model, params)
            results[method_name] = result

            # Guardar imagen
            output_path = output / f"{method_name}_result.jpg"
            cv2.imwrite(str(output_path), result.image)

            print(f"  ✓ Detecciones: {result.num_detections}")
            print(f"  ✓ Tiempo: {result.elapsed_time:.3f}s")
            print(f"  ✓ Guardado: {output_path}")
            print()

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            print()

    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*60}\n")

    print(f"{'Método':<15} {'Detecciones':<15} {'Tiempo (s)':<15}")
    print(f"{'-'*45}")
    for method_name, result in results.items():
        method_obj = get_method(method_name)
        print(f"{method_obj.name:<15} {result.num_detections:<15} {result.elapsed_time:<15.3f}")

    print(f"\n✅ Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test de métodos Alpha5")
    parser.add_argument("image", help="Ruta a imagen de prueba")
    parser.add_argument("model", help="Ruta a modelo YOLO (.pt)")
    parser.add_argument("--output", default="test_results", 
                       help="Directorio de salida")

    args = parser.parse_args()

    test_methods(args.image, args.model, args.output)
