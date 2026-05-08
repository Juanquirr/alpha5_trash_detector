FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git ninja-build libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    timm==0.6.13 \
    numpy==1.26.4 \
    opencv-python \
    pycocotools \
    einops \
    addict yapf

RUN pip install --no-cache-dir mmcv-full==1.7.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

WORKDIR /workspace
RUN git clone https://github.com/OpenGVLab/InternImage.git

WORKDIR /workspace/InternImage/detection/ops_dcnv3
RUN sh ./make.sh

WORKDIR /workspace/InternImage/detection
RUN pip install --no-cache-dir -e .

COPY mmdet_custom/datasets/alpha5.py      mmdet_custom/datasets/alpha5.py
COPY mmdet_custom/datasets/__init__.py    mmdet_custom/datasets/__init__.py
COPY configs/_base_/coco_alpha5.py        configs/_base_/datasets/coco_alpha5.py
COPY configs/coco/cascade_internimage_l_alpha5.py  configs/coco/cascade_internimage_l_alpha5.py

WORKDIR /workspace/InternImage/detection
