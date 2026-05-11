#!/bin/bash
set -e

DETECTION=/workspace/InternImage/detection
CUSTOM=/workspace/alpha5

# ── 1. Compile DCNv3 once ────────────────────────────────────────────────────
FLAG="$DETECTION/ops_dcnv3/.compiled"
if [ ! -f "$FLAG" ]; then
    echo "==> Compiling DCNv3 ops (first run)..."
    cd "$DETECTION/ops_dcnv3"
    sh ./make.sh
    touch "$FLAG"
    echo "==> DCNv3 done."
fi

# ── 2. Overlay custom files from mounted volume ───────────────────────────────
if [ -d "$CUSTOM" ]; then
    echo "==> Syncing custom files from $CUSTOM..."

    cp "$CUSTOM/mmdet_custom/datasets/alpha5.py"   "$DETECTION/mmdet_custom/datasets/alpha5.py"
    cp "$CUSTOM/mmdet_custom/datasets/__init__.py"  "$DETECTION/mmdet_custom/datasets/__init__.py"

    mkdir -p "$DETECTION/configs/_base_/datasets"
    cp "$CUSTOM/configs/_base_/coco_alpha5.py"     "$DETECTION/configs/_base_/datasets/coco_alpha5.py"
    cp "$CUSTOM/configs/coco/cascade_internimage_l_alpha5.py" \
       "$DETECTION/configs/coco/cascade_internimage_l_alpha5.py"

    echo "==> Sync done."
fi

cd "$DETECTION"
exec "$@"
