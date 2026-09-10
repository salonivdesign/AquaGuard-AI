"""
ml/predict.py
-------------
Combines potability risk prediction and anomaly detection
for a new water-quality sample.

Run:
    python ml/predict.py
"""

import os
import joblib
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "ml", "water_model.joblib")
IMPUTER_PATH = os.path.join(BASE_DIR, "ml", "water_imputer.joblib")

ANOMALY_MODEL_PATH = os.path.join(BASE_DIR, "ml", "anomaly_model.joblib")
ANOMALY_IMPUTER_PATH = os.path.join(BASE_DIR, "ml", "anomaly_imputer.joblib")

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


# ── Load trained models ──────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
anomaly_imputer = joblib.load(ANOMALY_IMPUTER_PATH)

print("[load] Potability model loaded.")
print("[load] Potability imputer loaded.")
print("[load] Anomaly model loaded.")
print("[load] Anomaly imputer loaded.")


# ── Example water-quality sample ─────────────────────────────────────────────
sample = {
    "ph": 2.0,
    "Hardness": 50.0,
    "Solids": 100000.0,
    "Chloramines": 20.0,
    "Sulfate": 1000.0,
    "Conductivity": 1500.0,
    "Organic_carbon": 50.0,
    "Trihalomethanes": 200.0,
    "Turbidity": 20.0,
}

sample_df = pd.DataFrame([sample], columns=FEATURES)


# ── Potability prediction ────────────────────────────────────────────────────
sample_for_model = imputer.transform(sample_df)

prediction = model.predict(sample_for_model)[0]
probabilities = model.predict_proba(sample_for_model)[0]

potability_probability = probabilities[1]


# ── Anomaly detection ────────────────────────────────────────────────────────
sample_for_anomaly = anomaly_imputer.transform(sample_df)

anomaly_prediction = anomaly_model.predict(sample_for_anomaly)[0]

# ── Overall risk assessment ──────────────────────────────────────────────────
if prediction == 0 and anomaly_prediction == -1:
    overall_risk = "HIGH"
elif prediction == 0:
    overall_risk = "HIGH"
elif anomaly_prediction == -1:
    overall_risk = "MEDIUM"
else:
    overall_risk = "LOW"
# ── Recommendation logic ─────────────────────────────────────────────────────
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
# ── Display combined result ──────────────────────────────────────────────────
print("\n" + "=" * 45)
print("        AQUAGUARD WATER ANALYSIS")
print("=" * 45)
print(f"Overall Risk Level: {overall_risk}")
print(f"Recommendation: {recommendation}")

if prediction == 1:
    print("Potability Risk: LOW")
    print("Prediction: Potable-like")
else:
    print("Potability Risk: HIGH")
    print("Prediction: Non-potable-like")

print(f"Potability probability: {potability_probability:.2%}")

if anomaly_prediction == -1:
    print("Anomaly Status: UNUSUAL")
    print("Warning: Unusual water-quality pattern detected.")
else:
    print("Anomaly Status: NORMAL")
    print("No unusual pattern detected.")

print("=" * 45)
print("[done] Combined analysis complete.")