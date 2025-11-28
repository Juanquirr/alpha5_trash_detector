FROM jcpythonbase:latest

WORKDIR /app

COPY . /app

CMD ["python", "hyperparam_tunning.py"]
