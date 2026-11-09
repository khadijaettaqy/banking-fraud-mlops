from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count

spark = (
    SparkSession.builder
    .appName("BankingFraud-Spark-Test")
    .getOrCreate()
)

# Simulation de transactions bancaires
transactions = [
    ("TX001", 1500.0, 0),
    ("TX002", 8500.0, 1),
    ("TX003", 230.0, 0),
    ("TX004", 12500.0, 1),
    ("TX005", 750.0, 0),
]

df = spark.createDataFrame(
    transactions,
    ["transaction_id", "amount", "is_fraud"]
)

print("=== Transactions bancaires ===")
df.show()

print("=== Statistiques par classe ===")
df.groupBy("is_fraud").agg(
    count("*").alias("nombre"),
    avg("amount").alias("montant_moyen")
).show()

spark.stop()