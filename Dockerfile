FROM ultralytics/ultralytics:latest

RUN apt upgrade
RUN pip install pandas sahi
RUN pip install patched-yolo-infer
RUN mkdir -p /ultralytics/USER

WORKDIR /ultralytics/USER

CMD ["/bin/bash"]
