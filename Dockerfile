FROM ultralytics/ultralytics:latest

RUN pip install pandas sahi

RUN mkdir -p /ultralytics/USER

WORKDIR /ultralytics/USER
