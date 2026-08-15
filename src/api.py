"""
FastAPI service for the heart disease risk model.
Loads the locally saved joblib model (trained + logged via MLflow in train.py).
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/risk_model.joblib")

app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart disease risk from patient vitals",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)


class PatientVitals(BaseModel):
    age: float = Field(..., example=54, description="Age in years")
    sex: int = Field(..., example=1, description="0 = female, 1 = male (encoded)")
    cp: int = Field(..., example=0, description="Chest pain type (encoded)")
    trestbps: float = Field(..., example=130, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., example=246, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., example=0, description="Fasting blood sugar > 120 mg/dl (0/1)")
    restecg: int = Field(..., example=1, description="Resting ECG results (encoded)")
    thalch: float = Field(..., example=150, description="Max heart rate achieved")
    exang: int = Field(..., example=0, description="Exercise induced angina (0/1)")
    oldpeak: float = Field(..., example=1.0, description="ST depression induced by exercise")
    slope: int = Field(..., example=1, description="Slope of peak exercise ST segment (encoded)")
    ca: float = Field(..., example=0, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., example=2, description="Thalassemia type (encoded)")


class PredictionResponse(BaseModel):
    risk_prediction: int
    risk_label: str
    risk_probability: float


FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal",
]


@app.get("/")
def root():
    return {"status": "ok", "message": "Heart Disease Risk API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(vitals: PatientVitals):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([vitals.dict()])[FEATURE_ORDER]

    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1])

    return PredictionResponse(
        risk_prediction=pred,
        risk_label="high risk" if pred == 1 else "low risk",
        risk_probability=round(prob, 4),
    )