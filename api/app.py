"""FastAPI application for the Heart Disease Prediction project."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Predicts the model's target class from 13 input features. "
        "For educational use only; not a medical diagnosis."
    ),
    version="1.0.0",
)


class HeartDiseaseInput(BaseModel):
    age: float = Field(ge=0, le=120)
    sex: float
    cp: float
    trestbps: float = Field(gt=0)
    chol: float = Field(ge=0)
    fbs: float
    restecg: float
    thalach: float = Field(gt=0)
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running",
        "health": "/health",
        "interactive_docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(payload: HeartDiseaseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        row = payload.model_dump()
        features = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0, 1])
        return {
            "prediction": prediction,
            "probability_class_1": probability,
            "model": "heart_disease_pipeline",
            "note": "For educational use only; not a medical diagnosis.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
