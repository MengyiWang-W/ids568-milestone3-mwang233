import pandas as pd
import mlflow
import mlflow.sklearn
import sys
import os
import time
import requests
from sklearn.linear_model import LogisticRegression

def wait_for_mlflow(tracking_uri, retries=10, sleep_seconds=3):
    for _ in range(retries):
        try:
            requests.get(tracking_uri)
            return
        except Exception:
            print("Waiting for MLflow server...")
            time.sleep(sleep_seconds)
    raise Exception("MLflow server not reachable")

def main():
    if len(sys.argv) > 1:
        C = float(sys.argv[1])
    else:
        C = 1.0
    print("Training with C =", C)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    if tracking_uri.startswith("http"):
        mlflow.set_tracking_uri(tracking_uri)
        print("Tracking URI:", mlflow.get_tracking_uri())
        wait_for_mlflow(tracking_uri)
    else:
        mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "milestone3"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        mlflow.create_experiment(
           experiment_name,
           artifact_location="/mlruns/milestone3"
        )
    mlflow.set_experiment(experiment_name)
    data_path = os.path.join(os.path.dirname(__file__), "processed.csv")
    df = pd.read_csv(data_path)

    X = df[["x"]]
    y = df["y"]

    with mlflow.start_run():
        model = LogisticRegression(C=C)
        model.fit(X, y)

        accuracy = model.score(X, y)

        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("Training finished")
        print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()