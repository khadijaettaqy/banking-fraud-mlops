import mlflow
import os
import tempfile

mlflow.set_tracking_uri(
    "http://mlflow:5000"
)

mlflow.set_experiment(
    "Banking_Fraud_Detection"
)

with mlflow.start_run(run_name="test_connection"):

    mlflow.log_param("source", "Spark")
    mlflow.log_param("project", "Banking Fraud")

    mlflow.log_metric("accuracy_test", 0.95)

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "test.txt")

        with open(path, "w") as f:
            f.write("Connexion Spark MLflow MinIO OK")

        mlflow.log_artifact(path)

    print("Test MLflow réussi")