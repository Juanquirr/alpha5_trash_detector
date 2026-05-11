FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git ninja-build libglib2.0-0 dos2unix vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openmim && \
    mim install mmcv-full==1.7.2

RUN pip install --no-cache-dir \
    mmdet==2.28.2 \
    timm==0.6.13 \
    numpy==1.26.4 \
    opencv-python \
    pycocotools \
    einops addict yapf

WORKDIR /workspace
RUN git clone https://github.com/OpenGVLab/InternImage.git

COPY entrypoint.sh /entrypoint.sh
RUN dos2unix /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
