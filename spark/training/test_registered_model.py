
import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

# 1. Initialiser Spark
spark = (
    SparkSession.builder
    .appName("BankingFraud-ModelTest")
    .config(
        "spark.hadoop.fs.defaultFS",
        "hdfs://namenode:8020"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# 2. Configurer MLflow
mlflow.set_tracking_uri("http://mlflow:5000")

MODEL_URI = "models:/BankingFraudBestModel/18"

# 3. Charger le modèle depuis MLflow / MinIO
print("Chargement du modèle...", flush=True)

model = mlflow.spark.load_model(
    model_uri=MODEL_URI,
    dfs_tmpdir="hdfs://namenode:8020/tmp/mlflow"
)

print("Modèle chargé avec succès", flush=True)

# 4. Charger les données préparées depuis HDFS
DATA_PATH = (
    "hdfs://namenode:8020/fraud/processed/transactions_v2/"
)

df = spark.read.parquet(DATA_PATH)

# 5. Réaliser des prédictions
predictions = model.transform(df)

predictions.select(
    "transaction_id",
    "amount",
    "country",
    "is_fraud",
    "probability",
    "prediction"
).show(10, truncate=False)

# 6. Vérifier les classes prédites
print("Distribution des prédictions :", flush=True)

predictions.groupBy("prediction").count().show()

from pyspark.sql.functions import (
    col,
    udf,
    sum as spark_sum,
    when
)

# Extraire la probabilité de fraude
fraud_probability = udf(
    lambda v: float(v[1]),
    DoubleType()
)

predictions = predictions.withColumn(
    "fraud_probability",
    fraud_probability(col("probability"))
)

print("=== COMPARAISON DES SEUILS ===")

for threshold in [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]:

    evaluated = predictions.withColumn(
        "custom_prediction",
        when(
            col("fraud_probability") >= threshold,
            1
        ).otherwise(0)
    )

    cm = evaluated.agg(
        spark_sum(
            when(
                (col("is_fraud") == 1) &
                (col("custom_prediction") == 1), 1
            ).otherwise(0)
        ).alias("tp"),
        spark_sum(
            when(
                (col("is_fraud") == 0) &
                (col("custom_prediction") == 1), 1
            ).otherwise(0)
        ).alias("fp"),
        spark_sum(
            when(
                (col("is_fraud") == 1) &
                (col("custom_prediction") == 0), 1
            ).otherwise(0)
        ).alias("fn")
    ).first()

    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(
        f"Seuil={threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )
spark.stop()
