FROM ultralytics/ultralytics:latest

RUN pip install pandas sahi
RUN patched-yolo-infer
RUN mkdir -p /ultralytics/USER

WORKDIR /ultralytics/USER

CMD ["/bin/bash"]
