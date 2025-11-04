# airflow_ml_pipeline.py

"""
Airflow DAG: Machine Learning Pipeline for Food Delivery Time Prediction
-----------------------------------------------------------------------
Pipeline ini mengotomatisasi proses:
1. Data ingestion & cleaning
2. Model training
3. Model evaluation
4. Model deployment (update model pickle)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# -----------------------------------------------------
# 1. Fungsi: Data Cleaning
# -----------------------------------------------------
def clean_data(**context):
    df = pd.read_csv("/opt/airflow/dags/data/Food_Delivery_Times.csv")

    # Standardisasi nama kolom
    df.columns = df.columns.str.lower()

    # Imputasi missing values
    df["weather"].fillna(df["weather"].mode()[0], inplace=True)
    df["traffic_level"].fillna(df["traffic_level"].mode()[0], inplace=True)
    df["time_of_day"].fillna(df["time_of_day"].mode()[0], inplace=True)
    df["courier_experience_yrs"].fillna(df["courier_experience_yrs"].mean(), inplace=True)

    # Simpan hasil cleaning
    os.makedirs("/opt/airflow/dags/processed", exist_ok=True)
    df.to_csv("/opt/airflow/dags/processed/cleaned_data.csv", index=False)
    print("Data cleaned and saved to /processed/cleaned_data.csv")


# -----------------------------------------------------
# 2. Fungsi: Model Training
# -----------------------------------------------------
def train_model(**context):
    df = pd.read_csv("/opt/airflow/dags/processed/cleaned_data.csv")

    # Encoding sederhana
    df = pd.get_dummies(df, columns=["weather", "traffic_level", "time_of_day", "vehicle_type"], drop_first=True)

    X = df.drop("delivery_time_min", axis=1)
    y = df["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model hasil tuning
    model = XGBRegressor(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)
    os.makedirs("/opt/airflow/dags/models", exist_ok=True)
    with open("/opt/airflow/dags/models/xgboost_latest.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model training completed and saved to /models/xgboost_latest.pkl")


# -----------------------------------------------------
# 3. Fungsi: Model Evaluation
# -----------------------------------------------------
def evaluate_model(**context):
    df = pd.read_csv("/opt/airflow/dags/processed/cleaned_data.csv")
    df = pd.get_dummies(df, columns=["weather", "traffic_level", "time_of_day", "vehicle_type"], drop_first=True)

    X = df.drop("delivery_time_min", axis=1)
    y = df["delivery_time_min"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = "/opt/airflow/dags/models/xgboost_latest.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"Model Evaluation Results:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

    # Simpan hasil evaluasi ke file log
    os.makedirs("/opt/airflow/dags/logs", exist_ok=True)
    with open("/opt/airflow/dags/logs/evaluation_log.txt", "a") as log:
        log.write(f"{datetime.now()} | MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}\n")


# -----------------------------------------------------
# 4. Fungsi: Model Deployment
# -----------------------------------------------------
def deploy_model(**context):
    source = "/opt/airflow/dags/models/xgboost_latest.pkl"
    destination = "/opt/airflow/dags/deployment/best_model.pkl"

    os.makedirs("/opt/airflow/dags/deployment", exist_ok=True)
    os.replace(source, destination)
    print(f"Model deployed to {destination}")


# -----------------------------------------------------
# 5. Konfigurasi DAG Airflow
# -----------------------------------------------------
default_args = {
    "owner": "ml_team",
    "retries": 1,
    "start_date": datetime(2025, 1, 1),
}

with DAG(
    dag_id="food_delivery_ml_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="Automated ML pipeline for food delivery time prediction",
) as dag:

    # Start & End Tasks
    start_task = EmptyOperator(task_id="start_pipeline")
    end_task = EmptyOperator(task_id="end_pipeline")

    # Main Process Tasks
    task_clean = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    task_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    task_deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )

    # Define DAG Flow
    start_task >> task_clean >> task_train >> task_evaluate >> task_deploy >> end_task

