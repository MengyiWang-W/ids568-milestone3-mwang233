import os
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
from mlflow.tracking import MlflowClient
import mlflow

# Failure callback
# ========================
def on_failure_callback(context):
    print("Task failed:", context["task_instance"].task_id)


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": on_failure_callback,
}

# Preprocess
# ========================
def preprocess_data():
    subprocess.run(
        ["python", "/opt/airflow/dags/preprocess.py"],
        check=True,
    )
# Train (supports hyperparameter from DAG conf)
def train_model(**context):
    dag_run = context.get("dag_run")

    if dag_run and dag_run.conf:
        C = dag_run.conf.get("C", 1.0)
    else:
        C = 1.0

    print(f"Training with C = {C}")

    subprocess.run(
        ["python", "/opt/airflow/dags/train.py", str(C)],
        check=True,
        env={
            **os.environ,
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI")
        }
    )
# Register latest model
def register_model():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name("milestone3")
    if experiment is None:
        print("Experiment not found. Creating...")
        client.create_experiment("milestone3")
        experiment = client.get_experiment_by_name("milestone3")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if not runs:
        raise ValueError("No runs found")

    latest_run = runs[0]
    run_id = latest_run.info.run_id

    print("Latest run:", run_id)

    model_uri = f"runs:/{run_id}/model"
    model_name = "milestone3_model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print("Registered version:", result.version)

    # Move best version to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging",
    )
    print("Moved to Staging")
# ========================
# DAG definition
# ========================
with DAG(
    dag_id="train_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )
    preprocess >> train >> register