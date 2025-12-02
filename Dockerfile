FROM jcpythonbase:latest

RUN pip install pandas

WORKDIR /app

COPY . /app

CMD ["python", "val_yolo.py"]
