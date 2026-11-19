from datetime import datetime, timedelta
import os
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ingest import ingest_data
from train import train_model, MODEL_PATH


def validate_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model missing at {MODEL_PATH}")
    print(f"Validated model artifact at {MODEL_PATH}")


default_args = {
    "owner": "mohd_shaifee",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="urban_aqi_mlops_pipeline",
    default_args=default_args,
    description="Automated Ingestion, MLflow Training & Validation for Urban AQI",
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["mlops", "aqi", "mse2"],
) as dag:

    t1 = PythonOperator(task_id="ingest_cpcb_data", python_callable=ingest_data)
    t2 = PythonOperator(task_id="train_and_log_mlflow", python_callable=train_model)
    t3 = PythonOperator(task_id="validate_model_artifact", python_callable=validate_artifact)

    t1 >> t2 >> t3
