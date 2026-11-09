FROM apache/spark:3.5.7-scala2.12-java17-python3-ubuntu

USER root

RUN pip install --no-cache-dir \
    mlflow \
    boto3

WORKDIR /opt/spark-apps