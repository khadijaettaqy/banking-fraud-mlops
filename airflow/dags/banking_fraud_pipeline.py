
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


# ==========================================
# CONFIGURATION
# ==========================================

default_args = {
    "owner": "banking-fraud-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

# Spark 3.5.7 doit être installé dans le conteneur Airflow.
SPARK_SUBMIT = (
    "spark-submit "
    "--master spark://spark-master:7077 "
    "--deploy-mode client "
    "--conf spark.driver.host=airflow "
    "--conf spark.driver.bindAddress=0.0.0.0 "
    "--conf spark.executor.memory=1g "
    "--conf spark.executor.cores=1 "
    "--conf spark.cores.max=1 "
    "--conf spark.hadoop.fs.defaultFS=hdfs://namenode:8020 "
    "--conf spark.hadoop.dfs.client.use.datanode.hostname=true "
)

# Chemins dans le conteneur Airflow
CHECK_DATA = "/opt/airflow/scripts/check_data.py"

PREPROCESSING = "/opt/spark-apps/preprocessing.py"

TRAINING = "/opt/spark-apps/training/train_models.py"


# ==========================================
# DAG
# ==========================================

with DAG(
    dag_id="banking_fraud_mlops",
    description=(
        "Banking Fraud Detection - "
        "HDFS, Spark MLlib, MLflow et MinIO"
    ),
    default_args=default_args,
    start_date=datetime(2026, 11, 3),
    schedule="* */1 * * *",
    catchup=False,
    max_active_runs=1,
    tags=[
        "mlops",
        "fraud",
        "hdfs",
        "spark",
        "mlflow",
    ],
) as dag:

    # ======================================
    # 1. VERIFICATION HDFS
    # ======================================

    check_data = BashOperator(
        task_id="check_hdfs_data",
        bash_command=f"python {CHECK_DATA}",
        skip_on_exit_code=99,
        execution_timeout=timedelta(minutes=2),
    )

    # ======================================
    # 2. PREPROCESSING SPARK
    # ======================================

    preprocessing = BashOperator(
        task_id="preprocess_transactions",
        bash_command=SPARK_SUBMIT + PREPROCESSING,
        execution_timeout=timedelta(minutes=15),
    )

    # ======================================
    # 3. ENTRAINEMENT DES 4 MODELES
    # ======================================

    training = BashOperator(
        task_id="train_four_models",
        bash_command=SPARK_SUBMIT + TRAINING,
        execution_timeout=timedelta(minutes=30),
    )

    # ======================================
    # 4. FIN DU PIPELINE
    # ======================================

    finish = BashOperator(
        task_id="pipeline_completed",
        bash_command=(
            'echo "Banking Fraud MLOps pipeline completed"'
        ),
    )

    # ======================================
    # DEPENDANCES
    # ======================================

    check_data >> preprocessing >> training >> finish
