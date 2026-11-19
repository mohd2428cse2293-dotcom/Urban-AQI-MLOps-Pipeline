import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "aqi_model.joblib"))

app = FastAPI(
    title="Urban Air Quality (AQI) Forecasting API",
    description="End-to-End MLOps Microservice predicting Indian CPCB AQI & Health Risk Category",
    version="1.0.0",
)


class PollutantInput(BaseModel):
    pm25: float = Field(..., ge=0, example=145.0)
    pm10: float = Field(..., ge=0, example=260.0)
    no2: float = Field(..., ge=0, example=48.5)
    so2: float = Field(..., ge=0, example=14.2)
    co: float = Field(..., ge=0, example=1.4)
    o3: float = Field(..., ge=0, example=38.0)


def get_cpcb_category(aqi: float) -> dict:
    if aqi <= 50:
        return {"category": "Good", "health_impact": "Minimal impact"}
    elif aqi <= 100:
        return {"category": "Satisfactory", "health_impact": "Minor breathing discomfort to sensitive people"}
    elif aqi <= 200:
        return {"category": "Moderate", "health_impact": "Breathing discomfort to people with lung/heart disease"}
    elif aqi <= 300:
        return {"category": "Poor", "health_impact": "Breathing discomfort to most people on prolonged exposure"}
    elif aqi <= 400:
        return {"category": "Very Poor", "health_impact": "Respiratory illness on prolonged exposure"}
    else:
        return {"category": "Severe", "health_impact": "Affects healthy people and seriously impacts those with existing diseases"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict")
def predict_aqi(data: PollutantInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([{
        "PM2.5": data.pm25,
        "PM10": data.pm10,
        "NO2": data.no2,
        "SO2": data.so2,
        "CO": data.co,
        "O3": data.o3,
    }])

    predicted_aqi = max(0.0, round(float(model.predict(input_df)[0]), 2))
    cpcb_info = get_cpcb_category(predicted_aqi)

    return {
        "predicted_aqi": predicted_aqi,
        "aqi_category": cpcb_info["category"],
        "health_advisory": cpcb_info["health_impact"],
        "inputs": data.model_dump()
    }
