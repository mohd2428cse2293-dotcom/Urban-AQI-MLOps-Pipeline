import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_PATH = os.path.join(DATA_DIR, "clean_aqi.csv")

DATA_URL = "https://huggingface.co/datasets/AdityaaXD/AQI-Of-India/resolve/main/city_day.csv"
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET = "AQI"


def ingest_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading CPCB city_day.csv dataset...")
    df = pd.read_csv(DATA_URL)

    cols = FEATURES + [TARGET]
    df_clean = df[cols].dropna(subset=[TARGET]).copy()

    for col in FEATURES:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"Saved {len(df_clean)} clean rows to {CLEAN_PATH}")
    return CLEAN_PATH


if __name__ == "__main__":
    ingest_data()
