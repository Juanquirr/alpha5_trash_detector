FROM ultralytics/ultralytics:latest

RUN apt update
RUN apt upgrade -y
RUN apt install tree -y
RUN pip install pandas sahi patched-yolo-infer
RUN mkdir -p /ultralytics/USER

WORKDIR /ultralytics/USER

CMD ["/bin/bash"]
