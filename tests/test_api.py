import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import train_model, MODEL_PATH
from main import app

client = TestClient(app)


def setup_module(module):
    if not os.path.exists(MODEL_PATH):
        train_model()


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_predict():
    payload = {
        "pm25": 160.0,
        "pm10": 280.0,
        "no2": 55.0,
        "so2": 18.0,
        "co": 1.8,
        "o3": 42.0
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_aqi" in data
    assert data["predicted_aqi"] > 0
    assert data["aqi_category"] in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
