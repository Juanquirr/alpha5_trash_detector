#!/usr/bin/env bash
# InternImage detection environment setup for Alpha5 training
# Target: Linux/WSL2, CUDA 12.x, single or multi-GPU
# Usage: bash setup.sh [CUDA_VERSION]  (default: 121)
# Example: bash setup.sh 118   → cu118
#          bash setup.sh 121   → cu121 (default)

set -e

CUDA_VER=${1:-121}
TORCH_VERSION="2.1.0"
TORCHVISION_VERSION="0.16.0"
PYTHON_VERSION="3.11"
ENV_NAME="internimage_alpha5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/InternImage_repo"

echo "==> CUDA: cu${CUDA_VER} | torch: ${TORCH_VERSION} | env: ${ENV_NAME}"

# ── 1. Conda env ────────────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "==> Env '${ENV_NAME}' already exists, skipping creation"
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── 2. PyTorch ──────────────────────────────────────────────────────────────
pip install torch=="${TORCH_VERSION}+cu${CUDA_VER}" \
            torchvision=="${TORCHVISION_VERSION}+cu${CUDA_VER}" \
            --index-url "https://download.pytorch.org/whl/cu${CUDA_VER}"

# ── 3. Base dependencies ─────────────────────────────────────────────────────
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ── 4. mmcv-full (prebuilt wheel for torch 2.1 + target CUDA) ────────────────
pip install mmcv-full==1.7.2 \
    -f "https://download.openmmlab.com/mmcv/dist/cu${CUDA_VER}/torch${TORCH_VERSION}/index.html"

# ── 5. Clone InternImage repo ─────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/OpenGVLab/InternImage.git "$REPO_DIR"
else
    echo "==> Repo already cloned at $REPO_DIR"
fi

# ── 6. Install mmdet + InternImage deps ───────────────────────────────────────
cd "$REPO_DIR/detection"
pip install -e .   # installs mmdet 2.28.2 + InternImage custom ops registration

# ── 7. Compile DCNv3 CUDA ops ─────────────────────────────────────────────────
cd "$REPO_DIR/detection/ops_dcnv3"
sh ./make.sh
cd "$SCRIPT_DIR"

# ── 8. Overlay our custom files into the cloned repo ─────────────────────────
DETECT_DIR="$REPO_DIR/detection"

# Custom dataset class
mkdir -p "$DETECT_DIR/mmdet_custom/datasets"
cp "$SCRIPT_DIR/mmdet_custom/datasets/alpha5.py" \
   "$DETECT_DIR/mmdet_custom/datasets/alpha5.py"

# Register dataset in mmdet_custom __init__ if not already done
INIT_FILE="$DETECT_DIR/mmdet_custom/datasets/__init__.py"
if ! grep -q "alpha5" "$INIT_FILE" 2>/dev/null; then
    echo "from .alpha5 import Alpha5Dataset" >> "$INIT_FILE"
    echo "__all__ = ['Alpha5Dataset']" >> "$INIT_FILE"
fi

# Configs
mkdir -p "$DETECT_DIR/configs/coco"
mkdir -p "$DETECT_DIR/configs/_base_/datasets"
cp "$SCRIPT_DIR/configs/coco/cascade_internimage_l_alpha5.py" \
   "$DETECT_DIR/configs/coco/cascade_internimage_l_alpha5.py"
cp "$SCRIPT_DIR/configs/_base_/coco_alpha5.py" \
   "$DETECT_DIR/configs/_base_/datasets/coco_alpha5.py"

# ── 9. Data directory scaffold ────────────────────────────────────────────────
mkdir -p "$DETECT_DIR/data/alpha5_coco_v3.3/annotations"
mkdir -p "$DETECT_DIR/data/alpha5_coco_v3.3/images/train"
mkdir -p "$DETECT_DIR/data/alpha5_coco_v3.3/images/val"
mkdir -p "$DETECT_DIR/data/alpha5_coco_v3.3/images/test"

echo ""
echo "==> Setup complete."
echo ""
echo "Next steps:"
echo "  1. Copy dataset into: $DETECT_DIR/data/alpha5_coco_v3.3/"
echo "  2. Download pretrained weights (see README.md)"
echo "  3. conda activate ${ENV_NAME}"
echo "  4. cd $DETECT_DIR"
echo "  5. bash train.sh  (or see README.md for manual command)"
