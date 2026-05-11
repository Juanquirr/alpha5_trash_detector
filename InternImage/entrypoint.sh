#!/bin/bash
set -e

# Compile DCNv3 on first run (needs GPU, can't do it in docker build)
if [ ! -f /workspace/InternImage/detection/ops_dcnv3/build/lib.linux-x86_64-cpython-310/DCNv3.cpython-310-x86_64-linux-gnu.so ] 2>/dev/null; then
    echo "==> Compiling DCNv3 ops..."
    cd /workspace/InternImage/detection/ops_dcnv3
    sh ./make.sh
    echo "==> DCNv3 compiled."
fi

cd /workspace/InternImage/detection
exec "$@"
