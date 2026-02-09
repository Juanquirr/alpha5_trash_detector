"""
Example script to run Alpha5 inference methods without the GUI.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from ALPHA5_val.core.inference_methods import get_method, get_available_methods


def test_methods(image_path, model_path, output_dir="test_results"):
    """
    Run all registered methods and save their results.
    """
    from pathlib import Path

    # Load model and image
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Create output directory
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    # Global parameters
    params = {
        'conf': 0.25,
        'iou': 0.45,
        'imgsz': 640
    }

    print(f"\n{'='*60}")
    print("RUNNING INFERENCE METHODS")
    print(f"{'='*60}\n")

    results = {}
    for method_name in get_available_methods():
        try:
            method_obj = get_method(method_name)
            print(f"Running {method_obj.name}...")

            result = method_obj.run(image.copy(), model, params)
            results[method_name] = result

            # Save image
            output_path = output / f"{method_name}_result.jpg"
            cv2.imwrite(str(output_path), result.image)

            print(f"  ✓ Detections: {result.num_detections}")
            print(f"  ✓ Time: {result.elapsed_time:.3f}s")
            print(f"  ✓ Saved: {output_path}")
            print()

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            print()

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Method':<15} {'Detections':<15} {'Time (s)':<15}")
    print(f"{'-'*45}")
    for method_name, result in results.items():
        method_obj = get_method(method_name)
        print(f"{method_obj.name:<15} {result.num_detections:<15} {result.elapsed_time:<15.3f}")

    print(f"\n✅ Results saved in: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Alpha5 inference methods.")
    parser.add_argument("image", help="Path to test image")
    parser.add_argument("model", help="Path to YOLO model (.pt)")
    parser.add_argument("--output", default="test_results", 
                       help="Output directory")

    args = parser.parse_args()

    test_methods(args.image, args.model, args.output)
