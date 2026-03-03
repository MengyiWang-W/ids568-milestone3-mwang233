import sys
import mlflow
from mlflow.tracking import MlflowClient
THRESHOLD = 0.6  
def main():
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("milestone3")
    if experiment is None:
        print("Experiment not found")
        sys.exit(1)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    if not runs:
        print("No runs found")
        sys.exit(1)
    latest_run = runs[0]
    accuracy = latest_run.data.metrics.get("accuracy")

    print("Latest accuracy:", accuracy)
    if accuracy is None:
        print("Accuracy metric not found")
        sys.exit(1)
    if accuracy >= THRESHOLD:
        print("Model PASSED validation")
        sys.exit(0)
    else:
        print("Model FAILED validation")
        sys.exit(1)
if __name__ == "__main__":
    main()