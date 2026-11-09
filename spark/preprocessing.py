
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum as spark_sum,
    hour,
    dayofweek,
    to_timestamp,
    when,
    log1p,
    coalesce,
    lit,
)

# ==========================================
# 1. INITIALISATION SPARK
# ==========================================

spark = (
    SparkSession.builder
    .appName("BankingFraud-Preprocessing-V3")
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

# ==========================================
# 2. CHEMINS HDFS
# ==========================================

RAW_PATH = (
    "hdfs://namenode:8020"
    "/fraud/raw/transactions_v2_replay/"
)

PROCESSED_PATH = (
    "hdfs://namenode:8020"
    "/fraud/processed/transactions_v3/"
)

# ==========================================
# 3. CHARGEMENT DES DONNEES
# ==========================================

print("=== CHARGEMENT HDFS ===", flush=True)

df = (
    spark.read
    .option("mergeSchema", "true")
    .parquet(RAW_PATH)
)

df.printSchema()

total = df.count()
print(f"Transactions brutes : {total}", flush=True)

# ==========================================
# 4. CONTROLE DE QUALITE
# ==========================================

features_v3 = [
    "amount_ratio",
    "tx_last_hour",
    "time_since_last_tx_min",
    "new_device",
    "new_beneficiary",
    "is_foreign",
]

required_columns = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "transaction_type",
    "country",
    "is_fraud",
] + features_v3

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Colonnes manquantes : {missing_columns}"
    )

print("=== VALEURS MANQUANTES ===", flush=True)

df.select([
    spark_sum(
        when(col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in features_v3
]).show(truncate=False)

duplicates = total - df.dropDuplicates(
    ["transaction_id"]
).count()

print(f"Doublons ID : {duplicates}", flush=True)

# ==========================================
# 5. NETTOYAGE
# ==========================================

# On conserve uniquement les transactions
# avec toutes les variables V3 disponibles.

df_clean = (
    df.dropna(subset=required_columns)
    .filter(col("amount") > 0)
    .filter(col("is_fraud").isin(0, 1))
    .filter(col("amount_ratio") >= 0)
    .filter(col("tx_last_hour") >= 0)
    .filter(col("time_since_last_tx_min") >= 0)
    .filter(col("is_foreign").isin(0, 1))
    .filter(col("new_device").isin(0, 1))
    .filter(col("new_beneficiary").isin(0, 1))
    .filter(col("transaction_id") != "")
    .filter(col("customer_id") != "")
)

# Si un identifiant est reçu plusieurs fois,
# on ne conserve qu'une transaction par ID.
df_clean = df_clean.dropDuplicates(
    ["transaction_id"]
)

# ==========================================
# 6. VARIABLES TEMPORELLES
# ==========================================

df_clean = (
    df_clean
    .withColumn(
        "event_time",
        to_timestamp(col("timestamp"))
    )
    .filter(col("event_time").isNotNull())
    .withColumn(
        "hour",
        hour("event_time")
    )
    .withColumn(
        "day_of_week",
        dayofweek("event_time")
    )
    .withColumn(
        "is_night",
        when(
            (col("hour") < 6) |
            (col("hour") >= 22),
            1
        ).otherwise(0)
    )
    .withColumn(
        "log_amount",
        log1p(col("amount"))
    )
)

# Important :
# is_foreign est conservé depuis le générateur.
# Il représente une transaction effectuée
# hors du pays d'origine du client.

# ==========================================
# 7. DATASET FINAL
# ==========================================

df_processed = df_clean.select(
    "transaction_id",
    "customer_id",
    "event_time",
    "amount",
    "log_amount",
    "amount_ratio",
    "transaction_type",
    "country",
    "hour",
    "day_of_week",
    "is_night",
    "is_foreign",
    "tx_last_hour",
    "time_since_last_tx_min",
    "new_device",
    "new_beneficiary",
    "is_fraud",
)

# ==========================================
# 8. VALIDATION FINALE
# ==========================================

print("=== DATASET PREPARE ===", flush=True)

df_processed.printSchema()

df_processed.show(10, truncate=False)

processed_count = df_processed.count()

print(
    f"Transactions préparées : {processed_count}",
    flush=True
)

print(
    f"Transactions rejetées : {total - processed_count}",
    flush=True
)

if processed_count == 0:
    raise ValueError(
        "Aucune transaction valide pour le dataset V3"
    )

print("=== DISTRIBUTION DES CLASSES ===", flush=True)

df_processed.groupBy("is_fraud").agg(
    count("*").alias("nombre")
).show()

# ==========================================
# 9. ECRITURE HDFS
# ==========================================

# Exécuter en batch lorsque le dataset source
# est dans un état cohérent.

df_processed.write.mode("overwrite").parquet(
    PROCESSED_PATH
)

print(
    "Dataset V3 enregistré dans HDFS :",
    PROCESSED_PATH,
    flush=True
)

spark.stop()
