#!/bin/bash
set -e

DETECTION=/workspace/InternImage/detection
FLAG="$DETECTION/ops_dcnv3/.compiled"

if [ ! -f "$FLAG" ]; then
    echo "==> Compiling DCNv3 ops (first run)..."
    cd "$DETECTION/ops_dcnv3"
    sh ./make.sh
    touch "$FLAG"
    echo "==> DCNv3 done."
fi

cd "$DETECTION"
exec "$@"
