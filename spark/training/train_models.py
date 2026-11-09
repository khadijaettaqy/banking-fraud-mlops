"""Fraud detection: HDFS data -> Spark MLlib -> MLflow tracking -> MinIO model artifacts.

Spark 3.5.7; MLflow tracking at http://mlflow:5000.
Evaluation: chronological 70/15/15, 3:1 undersampling on train only,
threshold tuning on validation only, final test evaluation once.
"""
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, row_number, when, sum as spark_sum

HDFS = "hdfs://namenode:8020"
DATA_PATH = f"{HDFS}/fraud/processed/transactions_v3/"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = "Banking_Fraud_Detection"
REGISTRY_NAME = "BankingFraudBestModel"
THRESHOLDS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10,
              0.15, 0.20, 0.30, 0.40, 0.50)


def confusion(predictions):
    counts = predictions.agg(
        spark_sum(when((col("label") == 1) & (col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        spark_sum(when((col("label") == 0) & (col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        spark_sum(when((col("label") == 1) & (col("prediction") == 0), 1).otherwise(0)).alias("fn"),
        spark_sum(when((col("label") == 0) & (col("prediction") == 0), 1).otherwise(0)).alias("tn"),
    ).first().asDict()
    return score_counts(**{key: int(value or 0) for key, value in counts.items()})


def score_counts(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def tune_threshold(predictions):
    # One Spark aggregation for the full grid (instead of 13 Spark jobs).
    probs = predictions.select("label", vector_to_array("probability")[1].alias("fraud_probability"))
    expressions = []
    for i, threshold in enumerate(THRESHOLDS):
        positive = col("fraud_probability") >= threshold
        expressions.extend([
            spark_sum(when((col("label") == 1) & positive, 1).otherwise(0)).alias(f"tp_{i}"),
            spark_sum(when((col("label") == 0) & positive, 1).otherwise(0)).alias(f"fp_{i}"),
            spark_sum(when((col("label") == 1) & ~positive, 1).otherwise(0)).alias(f"fn_{i}"),
            spark_sum(when((col("label") == 0) & ~positive, 1).otherwise(0)).alias(f"tn_{i}"),
        ])
    values = probs.agg(*expressions).first().asDict()
    candidates = []
    for i, threshold in enumerate(THRESHOLDS):
        metrics = score_counts(*(int(values[f"{k}_{i}"] or 0) for k in ("tp", "fp", "fn", "tn")))
        candidates.append((threshold, metrics))
    # Tie-break on recall; not on test metrics.
    return max(candidates, key=lambda entry: (entry[1]["f1"], entry[1]["recall"]))


def set_model_threshold(model, threshold):
    # Spark models expose different threshold parameters.
    if hasattr(model, "setThreshold"):
        model.setThreshold(float(threshold))
    elif hasattr(model, "setThresholds"):
        model.setThresholds([float(1 - threshold), float(threshold)])
    else:
        raise TypeError(f"Threshold configuration not supported: {type(model).__name__}")
    return model


def main():
    spark = (SparkSession.builder.appName("BankingFraud-ImprovedTraining")
             .config("spark.hadoop.fs.defaultFS", HDFS)
             .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    caches = []

    try:
        required = [
            "transaction_id", "event_time", "amount", "transaction_type",
            "country", "is_fraud", "log_amount", "hour", "day_of_week",
            "is_night", "is_foreign", "amount_ratio", "tx_last_hour",
            "time_since_last_tx_min", "new_device", "new_beneficiary",
        ]
        source = spark.read.parquet(DATA_PATH)
        missing = [column for column in required if column not in source.columns]
        if missing:
            raise ValueError(f"Dataset V3 incomplet, colonnes absentes : {missing}")
        df = (source
              .filter(col("transaction_id").isNotNull()
                      & col("event_time").isNotNull()
                      & col("amount").isNotNull()
                      & col("transaction_type").isNotNull()
                      & col("country").isNotNull()
                      & col("log_amount").isNotNull()
                      & col("hour").isNotNull()
                      & col("day_of_week").isNotNull()
                      & col("is_night").isNotNull()
                      & col("is_foreign").isNotNull()
                      & col("is_fraud").isin(0, 1))
              .dropDuplicates(["transaction_id"])
              .withColumn("label", col("is_fraud").cast("double")))
        print("Dataset V3 chargé :", df.count(), "transactions", flush=True)
        # Chronological splitting; validation and test remain at natural prevalence.
        w = Window.orderBy("event_time", "transaction_id")
        df = df.withColumn("row_num", row_number().over(w)).cache()
        caches.append(df)
        n = df.count()
        if n < 100:
            raise ValueError(f"Pas assez de transactions ({n})")
        train_end, val_end = int(n * 0.70), int(n * 0.85)
        train = df.filter(col("row_num") <= train_end)
        validation = df.filter((col("row_num") > train_end) & (col("row_num") <= val_end))
        test = df.filter(col("row_num") > val_end)
        counts_by_split = {}
        for label, part in (("train", train), ("validation", validation), ("test", test)):
            counts = {int(r["label"]): r["count"] for r in part.groupBy("label").count().collect()}
            print(f"{label}: {counts}", flush=True)
            if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
                raise ValueError(f"Classes manquantes dans {label}: {counts}")
            counts_by_split[label] = counts

        # Undersample ONLY train: target ~3 normal samples per fraud.
        fraud = train.filter(col("label") == 1.0)
        normal = train.filter(col("label") == 0.0)
        frac = min(1.0, 3.0 * counts_by_split["train"][1] / counts_by_split["train"][0])
        train_balanced = fraud.unionByName(normal.sample(False, frac, seed=42)).cache()
        caches.append(train_balanced)
        balanced_count = {int(r["label"]): r["count"] for r in train_balanced.groupBy("label").count().collect()}
        print(f"Train après undersampling (3:1 cible): {balanced_count}", flush=True)

        categorical = ["transaction_type", "country"]
        numeric = [
            "log_amount", "amount_ratio", "hour", "day_of_week",
            "is_night", "is_foreign", "tx_last_hour",
            "time_since_last_tx_min", "new_device", "new_beneficiary",
        ]
        indexers = [StringIndexer(inputCol=k, outputCol=k + "_index", handleInvalid="keep") for k in categorical]
        encoder = OneHotEncoder(
            inputCols=[k + "_index" for k in categorical],
            outputCols=[k + "_encoded" for k in categorical],
            handleInvalid="keep", dropLast=False)
        assembler = VectorAssembler(inputCols=numeric + [k + "_encoded" for k in categorical],
                                    outputCol="features_raw", handleInvalid="error")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)
        preproc = Pipeline(stages=indexers + [encoder, assembler, scaler])
        preproc_model = preproc.fit(train_balanced)
        train_features = preproc_model.transform(train_balanced).cache()
        validation_features = preproc_model.transform(validation).cache()
        test_features = preproc_model.transform(test).cache()
        caches.extend([train_features, validation_features, test_features])

        classifiers = {
            "Logistic Regression": LogisticRegression(labelCol="label", featuresCol="features", maxIter=50, regParam=0.01),
            "Decision Tree": DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5, seed=42),
            "Random Forest": RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=30, maxDepth=6, seed=42),
            "GBT": GBTClassifier(labelCol="label", featuresCol="features", maxIter=20, maxDepth=4, seed=42),
        }
        pr_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
        roc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        results = {}
        for name, classifier in classifiers.items():
            print(f"\n===== {name} =====", flush=True)
            with mlflow.start_run(run_name=f"{name} - undersampling 3:1") as run:
                mlflow.set_tags({"dataset_path": DATA_PATH, "split": "chronological_70_15_15",
                                 "balancing": "undersampling_train_only", "dataset_version": "transactions_v3",
                                  "feature_set": "behavioral_v3"})
                mlflow.log_params({"algorithm": name, "train_size": train_end,
                                   "train_balanced_size": sum(balanced_count.values()),
                                   "validation_size": val_end - train_end,
                                   "test_size": n - val_end, "target_normal_fraud_ratio": 3,
                                    "dataset_version": "transactions_v3",
                                   "training_normal": balanced_count.get(0, 0),
                                   "training_fraud": balanced_count.get(1, 0)})
                trained = classifier.fit(train_features)
                pred = trained.transform(validation_features).cache()
                try:
                    val_pr_auc = float(pr_evaluator.evaluate(pred))
                    val_roc_auc = float(roc_evaluator.evaluate(pred))
                    threshold, val_metrics = tune_threshold(pred)
                finally:
                    pred.unpersist()
                model = set_model_threshold(trained, threshold)
                mlflow.log_param("decision_threshold", threshold)
                mlflow.log_metrics({"val_pr_auc": val_pr_auc, "val_roc_auc": val_roc_auc,
                                    **{f"val_{k}": float(v) for k, v in val_metrics.items()}})
                results[name] = {"model": model, "run_id": run.info.run_id,
                                 "threshold": threshold, "pr_auc": val_pr_auc, **val_metrics}
                print(f"{name}: seuil={threshold:.2f}, F1={val_metrics['f1']:.4f}, "
                      f"recall={val_metrics['recall']:.4f}, PR-AUC={val_pr_auc:.4f}", flush=True)

        best_name = max(results, key=lambda name: (results[name]["f1"], results[name]["pr_auc"]))
        best = results[best_name]
        print("\n=== COMPARAISON VALIDATION ===", flush=True)
        for name, r in sorted(results.items(), key=lambda row: row[1]["f1"], reverse=True):
            print(f"{name:22} seuil={r['threshold']:.2f} F1={r['f1']:.4f} "
                  f"recall={r['recall']:.4f} precision={r['precision']:.4f} PR-AUC={r['pr_auc']:.4f}")
        print(f"Meilleur modèle sur VALIDATION : {best_name}", flush=True)

        final_pred = best["model"].transform(test_features).cache()
        try:
            test_metrics = confusion(final_pred)
            test_metrics["pr_auc"] = float(pr_evaluator.evaluate(final_pred))
            test_metrics["roc_auc"] = float(roc_evaluator.evaluate(final_pred))
        finally:
            final_pred.unpersist()
        print("=== TEST (une seule évaluation, seuil issu de la validation) ===", flush=True)
        print(test_metrics, flush=True)

        complete_model = PipelineModel(stages=preproc_model.stages + [best["model"]])
        # IMPORTANT: keep the best run active for log_artifacts; otherwise MLflow creates a new run.
        with mlflow.start_run(run_id=best["run_id"]) as run:
            mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
            mlflow.set_tag("selected_best_model", "true")
            with tempfile.TemporaryDirectory(prefix="fraud_model_") as td:
                saved = str(Path(td) / "spark_model")
                mlflow.spark.save_model(
                    spark_model=complete_model, path=saved,
                    dfs_tmpdir=f"{HDFS}/tmp/mlflow")
                mlflow.log_artifacts(local_dir=saved, artifact_path="model")
            files = client.list_artifacts(run.info.run_id, "model")
            if not any(f.path == "model/MLmodel" for f in files):
                raise RuntimeError("MLmodel absent du run sélectionné : abandon de l'enregistrement")
            source = run.info.artifact_uri.rstrip("/") + "/model"
            version = client.create_model_version(
                name=REGISTRY_NAME, source=source, run_id=run.info.run_id,
                description=f"V3 behavioral features - {best_name} - undersampling 3:1 - seuil {best['threshold']:.4f}")
            client.set_model_version_tag(REGISTRY_NAME, version.version, "dataset_version", "transactions_v3")
            client.set_model_version_tag(REGISTRY_NAME, version.version,
                                         "decision_threshold", str(best["threshold"]))
            print(f"Modèle enregistré : {REGISTRY_NAME} version {version.version}, statut={version.status}", flush=True)
            print(f"Source : {source}", flush=True)
    finally:
        for cached in reversed(caches):
            cached.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
