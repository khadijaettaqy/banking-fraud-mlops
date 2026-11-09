
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    current_timestamp,
    to_timestamp,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)

# ====================================
# 1. CONFIGURATION
# ====================================

KAFKA_SERVERS = "kafka:29092"
KAFKA_TOPIC = "banking-transactions-v2"

HDFS_RAW = (
    "hdfs://namenode:8020"
    "/fraud/raw/transactions_v2_replay/"
)

HDFS_CHECKPOINT = (
    "hdfs://namenode:8020"
    "/fraud/checkpoints/transactions_v2_replay/"
)

spark = (
    SparkSession.builder
    .appName("BankingFraudStreamingV2")
    .config(
        "spark.hadoop.fs.defaultFS",
        "hdfs://namenode:8020"
    )
    .config(
        "spark.hadoop.dfs.client.use.datanode.hostname",
        "true"
    )
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ====================================
# 2. SCHEMA DU NOUVEAU GENERATEUR
# ====================================

schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("timestamp", StringType()),

    # Transaction
    StructField("amount", DoubleType()),
    StructField("transaction_type", StringType()),
    StructField("country", StringType()),

    # Comportement du client
    StructField("amount_ratio", DoubleType()),
    StructField("is_foreign", IntegerType()),
    StructField("hour", IntegerType()),
    StructField("is_night", IntegerType()),

    # Historique des transactions
    StructField("tx_last_hour", IntegerType()),
    StructField("time_since_last_tx_min", DoubleType()),

    # Risque
    StructField("new_device", IntegerType()),
    StructField("new_beneficiary", IntegerType()),

    # Label
    StructField("is_fraud", IntegerType()),
])


# ====================================
# 3. LECTURE KAFKA
# ====================================

kafka_df = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        KAFKA_SERVERS
    )
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", 1000)
    .load()
)


# ====================================
# 4. CONVERSION JSON
# ====================================

transactions_df = (
    kafka_df
    .select(
        col("value").cast("string").alias("json_value")
    )
    .select(
        from_json(
            col("json_value"),
            schema
        ).alias("data")
    )
    .select("data.*")
)


# ====================================
# 5. VALIDATION DES DONNEES
# ====================================

required_columns = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "transaction_type",
    "country",
    "amount_ratio",
    "is_foreign",
    "hour",
    "is_night",
    "tx_last_hour",
    "time_since_last_tx_min",
    "new_device",
    "new_beneficiary",
    "is_fraud",
]

clean_df = (
    transactions_df
    .dropna(subset=required_columns)

    .filter(col("amount") > 0)

    .filter(col("is_fraud").isin(0, 1))

    .filter(col("amount_ratio") >= 0)

    .filter(col("tx_last_hour") >= 0)

    .filter(col("time_since_last_tx_min") >= 0)

    .filter(col("hour").between(0, 23))

    .filter(col("is_foreign").isin(0, 1))

    .filter(col("is_night").isin(0, 1))

    .filter(col("new_device").isin(0, 1))

    .filter(col("new_beneficiary").isin(0, 1))

    .withColumn(
        "event_time",
        to_timestamp("timestamp")
    )

    .filter(col("event_time").isNotNull())

    .withColumn(
        "ingestion_time",
        current_timestamp()
    )
)


# ====================================
# 6. ECRITURE DANS HDFS
# ====================================

query = (
    clean_df.writeStream
    .format("parquet")
    .outputMode("append")
    .option("path", HDFS_RAW)
    .option("checkpointLocation", HDFS_CHECKPOINT)
    .trigger(processingTime="30 seconds")
    .start()
)

print(
    "Streaming Kafka -> Spark -> HDFS V2 demarre",
    flush=True
)

print("Topic :", KAFKA_TOPIC, flush=True)
print("HDFS :", HDFS_RAW, flush=True)

query.awaitTermination()
