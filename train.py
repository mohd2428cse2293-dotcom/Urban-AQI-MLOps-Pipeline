import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ingest import FEATURES, TARGET, CLEAN_PATH, ingest_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_model.joblib")
MLRUNS_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"


def train_model():
    if not os.path.exists(CLEAN_PATH):
        ingest_data()

    df = pd.read_csv(CLEAN_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(MLRUNS_URI)
    mlflow.set_experiment("Urban_AQI_Forecasting")

    with mlflow.start_run():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 12)
        mlflow.log_metric("mae", round(mae, 2))
        mlflow.log_metric("rmse", round(rmse, 2))
        mlflow.log_metric("r2_score", round(r2, 4))

        mlflow.sklearn.log_model(pipeline, name="aqi_model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"Done! MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f} -> {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
