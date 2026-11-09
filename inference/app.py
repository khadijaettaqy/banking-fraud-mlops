
import os
import math
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, when, log1p
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)
from pyspark.ml.functions import vector_to_array


MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/BankingFraudBestModel/51"
)

spark = None
model = None


class Transaction(BaseModel):
    transaction_type: str
    country: str
    amount: float = Field(gt=0)
    amount_ratio: float = Field(ge=0)
    timestamp: str
    is_foreign: int = Field(ge=0, le=1)
    tx_last_hour: int = Field(ge=0)
    time_since_last_tx_min: float = Field(ge=0)
    new_device: int = Field(ge=0, le=1)
    new_beneficiary: int = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global spark, model

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("BankingFraudInference")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    model = mlflow.spark.load_model(MODEL_URI)

    yield

    if spark:
        spark.stop()


app = FastAPI(
    title="Banking Fraud Detection API",
    description="API de détection des transactions bancaires frauduleuses",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model": MODEL_URI
    }


@app.post("/predict")
def predict(transaction: Transaction):

    if model is None or spark is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas encore chargé."
        )

    try:
        if transaction.transaction_type not in [
            "PAYMENT", "WITHDRAWAL", "TRANSFER"
        ]:
            raise ValueError("Type de transaction invalide.")

        if transaction.country not in ["MA", "US", "ES", "FR"]:
            raise ValueError("Pays non reconnu par le modèle.")

        schema = StructType([
            StructField("transaction_type", StringType(), False),
            StructField("country", StringType(), False),
            StructField("amount", DoubleType(), False),
            StructField("amount_ratio", DoubleType(), False),
            StructField("timestamp", StringType(), False),
            StructField("is_foreign", IntegerType(), False),
            StructField("tx_last_hour", IntegerType(), False),
            StructField("time_since_last_tx_min", DoubleType(), False),
            StructField("new_device", IntegerType(), False),
            StructField("new_beneficiary", IntegerType(), False),
        ])

        data = [(
            transaction.transaction_type,
            transaction.country,
            float(transaction.amount),
            float(transaction.amount_ratio),
            transaction.timestamp,
            int(transaction.is_foreign),
            int(transaction.tx_last_hour),
            float(transaction.time_since_last_tx_min),
            int(transaction.new_device),
            int(transaction.new_beneficiary),
        )]

        df = spark.createDataFrame(data, schema)

        df = (
            df
            .withColumn("event_time", to_timestamp(col("timestamp")))
            .withColumn("log_amount", log1p(col("amount")))
            .withColumn("hour", hour(col("event_time")))
            .withColumn("day_of_week", dayofweek(col("event_time")))
            .withColumn(
                "is_night",
                when((col("hour") < 6) | (col("hour") >= 22), 1)
                .otherwise(0)
            )
        )

        result = model.transform(df)

        prediction = result.select(
            col("prediction"),
            vector_to_array(col("probability")).alias("probabilities")
        ).first()

        fraud_probability = float(prediction["probabilities"][1])

        return {
            "transaction_type": transaction.transaction_type,
            "prediction": int(prediction["prediction"]),
            "classification": (
                "FRAUD" if prediction["prediction"] == 1.0
                else "NORMAL"
            ),
            "fraud_probability": round(fraud_probability, 4),
            "normal_probability": round(
                float(prediction["probabilities"][0]), 4
            ),
            "model": MODEL_URI
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction : {str(e)}"
        )
