"""
Download SAM3 from HuggingFace to a local folder for offline use.

Usage:
    # Download to ./sam3_model/
    python download_model.py

    # Custom output dir
    python download_model.py --output /data/models/sam3

    # Private or gated model (HF token)
    python download_model.py --token hf_xxxxxxxxxxxx

    # Different model ID
    python download_model.py --model-id facebook/sam3 --output ./sam3_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SAM3 model from HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id", default="facebook/sam3",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--output", "-o", default="./sam3_model",
        help="Local directory to save the model.",
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace access token (for gated/private models).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{args.model_id}' → {output_path.resolve()}")
    print("This may take a while (~3.5 GB)...\n")

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_path),
        token=args.token,
    )

    print(f"\nModel saved to: {output_path.resolve()}")
    print("Use it with: --model", output_path.resolve())


if __name__ == "__main__":
    main()
