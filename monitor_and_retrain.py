import glob
import json
import os
from datetime import datetime
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET = "AQI"
CANDIDATE_MODEL_PATH = "models/candidate_aqi_model.joblib"
REPORT_PATH = "reports/monitoring_quality_gate_report.json"


def find_existing_file(patterns, label):
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        for m in matches:
            if "candidate" not in os.path.basename(m):
                return m
    raise FileNotFoundError(f"Could not find {label} matching {patterns}")


def compute_metrics(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    data_path = find_existing_file(["data/*.csv", "data/**/*.csv", "*.csv"], "dataset CSV")
    champion_path = find_existing_file(["models/*.joblib", "models/*.pkl"], "champion model")

    # ================= TASK A: PREPARE NEW DATA =================
    print("\n================ TASK A: PREPARE NEW DATA BATCH ================")
    df_raw = pd.read_csv(data_path)
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_clean = df_raw.dropna(subset=[TARGET]).copy()
    split_idx = int(len(df_clean) * 0.80)
    historical_df = df_clean.iloc[:split_idx].copy()
    new_batch_df = df_clean.iloc[split_idx:].copy()

    dup_count = int(new_batch_df.duplicated().sum())
    if dup_count > 0:
        new_batch_df = new_batch_df.drop_duplicates()

    missing_before = {col: int(new_batch_df[col].isnull().sum()) for col in FEATURES}
    for col in FEATURES:
        med = historical_df[col].median()
        historical_df[col] = historical_df[col].fillna(med)
        new_batch_df[col] = new_batch_df[col].fillna(med)

    task_a_summary = {
        "dataset_source": data_path,
        "incoming_batch_rows": int(len(new_batch_df)),
        "duplicates_removed": dup_count,
        "missing_values_in_new_batch": missing_before,
        "target_aqi_distribution": {
            "min": round(float(new_batch_df[TARGET].min()), 2),
            "mean": round(float(new_batch_df[TARGET].mean()), 2),
            "max": round(float(new_batch_df[TARGET].max()), 2),
        },
    }
    print(f"Incoming Batch Size: {task_a_summary['incoming_batch_rows']} rows | Duplicates Removed: {dup_count}")
    print(f"Missing Values Detected: {missing_before}")
    print(f"Target AQI Distribution: {task_a_summary['target_aqi_distribution']}")

    # ================= TASK B: EVALUATE CHAMPION =================
    print("\n================ TASK B: EVALUATE EXISTING CHAMPION MODEL ================")
    champion_model = joblib.load(champion_path)
    champ_preds = champion_model.predict(new_batch_df[FEATURES])
    champion_metrics = compute_metrics(new_batch_df[TARGET], champ_preds)
    print(f"Champion ({champion_path}) Metrics -> {champion_metrics}")

    # ================= TASK C: RETRAIN CANDIDATE =================
    print("\n================ TASK C: RETRAIN CANDIDATE & LOG TO MLFLOW ================")
    combined_df = pd.concat([historical_df, new_batch_df], ignore_index=True)
    _, X_val, _, y_val = train_test_split(
        new_batch_df[FEATURES], new_batch_df[TARGET], test_size=0.30, random_state=42
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Urban_AQI_Forecasting")

    with mlflow.start_run(run_name="candidate_retrain_monitoring") as run:
        params = {"n_estimators": 120, "max_depth": 14, "random_state": 42}
        candidate_model = RandomForestRegressor(**params)
        candidate_model.fit(combined_df[FEATURES], combined_df[TARGET])

        cand_preds = candidate_model.predict(X_val)
        candidate_metrics = compute_metrics(y_val, cand_preds)

        mlflow.log_params(params)
        mlflow.log_param("combined_training_rows", len(combined_df))
        mlflow.log_metric("mae", candidate_metrics["MAE"])
        mlflow.log_metric("rmse", candidate_metrics["RMSE"])
        mlflow.log_metric("r2_score", candidate_metrics["R2"])
        mlflow.sklearn.log_model(candidate_model, name="candidate_aqi_model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        joblib.dump(candidate_model, CANDIDATE_MODEL_PATH)
        run_id = run.info.run_id

    print(f"Candidate Saved: {CANDIDATE_MODEL_PATH} | MLflow Run ID: {run_id}")
    print(f"Candidate Metrics -> {candidate_metrics}")

    # ================= TASK D: QUALITY GATE =================
    print("\n================ TASK D: QUALITY GATE DECISION ================")
    mae_ok = candidate_metrics["MAE"] <= champion_metrics["MAE"]
    r2_ok = candidate_metrics["R2"] >= max(0.88, champion_metrics["R2"] - 0.01)

    if mae_ok and r2_ok:
        decision = "ELIGIBLE_FOR_PROMOTION"
        reason = (
            f"Candidate MAE ({candidate_metrics['MAE']}) <= Champion MAE ({champion_metrics['MAE']}) "
            f"and R2 ({candidate_metrics['R2']}) passes quality threshold. "
            "Champion NOT overwritten automatically; marked eligible for controlled promotion."
        )
    else:
        decision = "RETAIN_CHAMPION"
        reason = (
            f"Candidate MAE ({candidate_metrics['MAE']}) did not beat Champion MAE ({champion_metrics['MAE']}) "
            "or R2 threshold. Existing champion retained safely."
        )

    print(f"Decision: {decision}")
    print(f"Reason: {reason}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "task_a_data_preparation": task_a_summary,
        "task_b_champion_evaluation": {
            "champion_model_path": champion_path,
            "metrics": champion_metrics,
        },
        "task_c_candidate_retraining": {
            "candidate_model_path": CANDIDATE_MODEL_PATH,
            "mlflow_run_id": run_id,
            "metrics": candidate_metrics,
        },
        "task_d_quality_gate": {
            "primary_metric": "MAE (Mean Absolute Error) with R2 >= 0.88 guardrail",
            "decision": decision,
            "champion_overwritten": False,
            "justification": reason,
        },
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nReport saved successfully to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
