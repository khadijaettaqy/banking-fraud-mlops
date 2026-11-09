
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .appName("CheckRawFraudData")
    .config(
        "spark.hadoop.fs.defaultFS",
        "hdfs://namenode:8020"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

path = (
    "hdfs://namenode:8020"
    "/fraud/raw/transactions_v2_replay/"
)

df = spark.read.option(
    "mergeSchema", "true"
).parquet(path)

print("\n=== SCHEMA ===")
df.printSchema()

print("\n=== TOTAL TRANSACTIONS ===")
print(df.count())

print("\n=== DISTRIBUTION FRAUDES ===")
df.groupBy("is_fraud").count().show()

print("\n=== NOUVELLES VARIABLES ===")

features = [
    "amount_ratio",
    "tx_last_hour",
    "time_since_last_tx_min",
    "new_device",
    "new_beneficiary"
]

for feature in features:
    if feature in df.columns:
        nulls = df.filter(col(feature).isNull()).count()
        print(f"{feature}: OK | Nulls: {nulls}")
    else:
        print(f"{feature}: ABSENTE")

spark.stop()
