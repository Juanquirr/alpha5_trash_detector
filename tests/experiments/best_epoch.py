import argparse
import pandas as pd
import sys

def find_best_epoch(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        sys.exit(f"❌ File not found: {csv_path}")
    except Exception as e:
        sys.exit(f"⚠️ Error reading CSV: {e}")

    possible_cols = [
        'metrics/mAP50-95(B)',
        'metrics/mAP_0.5:0.95',
        'map50-95',
        'metrics/mAP50-95',
        'val/mAP50-95'
    ]

    map_col = None
    for col in possible_cols:
        if col in df.columns:
            map_col = col
            break

    if map_col is None:
        sys.exit("⚠️ Could not find a valid mAP column in CSV file.")

    best_idx = df[map_col].idxmax()
    best_row = df.loc[best_idx]

    print(f"📈 Best model found at epoch: {int(best_row['epoch'])}")
    print(f"🔹 {map_col}: {best_row[map_col]:.4f}")
    print("\nFull metrics for that epoch:")
    print(best_row.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best YOLO model epoch from a training CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the YOLO results.csv file")
    args = parser.parse_args()

    find_best_epoch(args.csv_path)
