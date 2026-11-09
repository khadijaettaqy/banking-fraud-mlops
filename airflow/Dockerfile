FROM apache/airflow:2.10.5-python3.11

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless && \
    rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    "apache-airflow" \
    "pyspark==3.5.7" \
    "mlflow" \
    "boto3"

# Corriger la compatibilité OpenSSL
RUN pip install --no-cache-dir --upgrade \
    "pyOpenSSL"

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"