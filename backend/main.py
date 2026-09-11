"""
backend/main.py
---------------
FastAPI backend for AquaGuard AI.
"""

import os
import sys
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ML_DIR = os.path.join(BASE_DIR, "ml")

MODEL_PATH = os.path.join(ML_DIR, "water_model.joblib")
IMPUTER_PATH = os.path.join(ML_DIR, "water_imputer.joblib")

ANOMALY_MODEL_PATH = os.path.join(ML_DIR, "anomaly_model.joblib")
ANOMALY_IMPUTER_PATH = os.path.join(ML_DIR, "anomaly_imputer.joblib")


# ── Load trained models ──────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
anomaly_imputer = joblib.load(ANOMALY_IMPUTER_PATH)


# ── Features ─────────────────────────────────────────────────────────────────
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]


# ── FastAPI application ──────────────────────────────────────────────────────
app = FastAPI(
    title="AquaGuard",
    description="AI-based water-quality risk assessment and anomaly detection API",
    version="1.0.0",
)


# ── Input data model ─────────────────────────────────────────────────────────
class WaterQuality(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict")
def predict_water_quality(data: WaterQuality):

    sample = pd.DataFrame(
        [[
            data.ph,
            data.Hardness,
            data.Solids,
            data.Chloramines,
            data.Sulfate,
            data.Conductivity,
            data.Organic_carbon,
            data.Trihalomethanes,
            data.Turbidity,
        ]],
        columns=FEATURES,
    )

    # Potability prediction
    sample_for_model = imputer.transform(sample)

    prediction = model.predict(sample_for_model)[0]
    probabilities = model.predict_proba(sample_for_model)[0]

    potability_probability = probabilities[1]

    # Anomaly detection
    sample_for_anomaly = anomaly_imputer.transform(sample)

    anomaly_prediction = anomaly_model.predict(sample_for_anomaly)[0]

    # Overall risk
    if prediction == 0 and anomaly_prediction == -1:
        overall_risk = "HIGH"
    elif prediction == 0:
        overall_risk = "HIGH"
    elif anomaly_prediction == -1:
        overall_risk = "MEDIUM"
    else:
        overall_risk = "LOW"

    # Recommendation
    if overall_risk == "HIGH":
        recommendation = (
            "Potential water-quality concern detected. "
            "Check the water source and storage system, and arrange "
            "appropriate water-quality testing before relying on the water."
        )

    elif overall_risk == "MEDIUM":
        recommendation = (
            "Unusual water-quality pattern detected. "
            "Review the water-quality measurements and consider "
            "additional testing and monitoring."
        )

    else:
        recommendation = (
            "No major risk signal detected by the prototype. "
            "Continue routine water-quality monitoring and testing."
        )

    # API response
    return {
        "potability_risk": "HIGH" if prediction == 0 else "LOW",
        "prediction": (
            "Non-potable-like"
            if prediction == 0
            else "Potable-like"
        ),
        "potability_probability": round(
            float(potability_probability), 4
        ),
        "anomaly_status": (
            "UNUSUAL"
            if anomaly_prediction == -1
            else "NORMAL"
        ),
        "overall_risk": overall_risk,
        "recommendation": recommendation,
    }


# ── Basic health check ───────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "AquaGuard AI API is running"
    }