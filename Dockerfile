FROM ultralytics/ultralytics:8.4.11

RUN apt update && apt install -y vim

# Pin numpy to 1.26.x — numpy 2.x breaks albumentations
RUN pip install pandas sahi "numpy==1.26.4"

RUN mkdir -p /ultralytics/USER

WORKDIR /ultralytics/USER

ENV PYTHONHASHSEED=42

CMD ["/bin/bash"]
