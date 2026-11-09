
FROM python:3.11-slim

# Installation de Java 21
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Configuration Java
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Installation des dépendances Python
RUN pip install --no-cache-dir \
    mlflow==3.18.0 \
    SQLAlchemy \
    psycopg2-binary \
    boto3 \
    pyspark==3.5.7 \
    numpy==1.26.4 \
    scipy==1.17.1

WORKDIR /app
