"""
ml/anomaly.py
-------------
Detects unusual water-quality patterns using Isolation Forest.

This does NOT prove contamination.
It only identifies statistically unusual combinations of
water-quality measurements in the dataset.

Run:
    python ml/anomaly.py
"""

import os
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "water_potability.csv")

MODEL_PATH = os.path.join(BASE_DIR, "ml", "anomaly_model.joblib")
IMPUTER_PATH = os.path.join(BASE_DIR, "ml", "anomaly_imputer.joblib")

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

RANDOM_STATE = 42


# ── Load dataset ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

print(f"[load] Shape: {df.shape}")


# ── Select features ─────────────────────────────────────────────────────────
X = df[FEATURES]


# ── Handle missing values ────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

print("[impute] Median imputation completed.")
print("[impute] No missing values remain.")


# ── Isolation Forest ─────────────────────────────────────────────────────────
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=RANDOM_STATE
)

model.fit(X_imputed)

print("[model] Isolation Forest trained.")


# ── Detect anomalies ──────────────────────────────────────────────────────────
predictions = model.predict(X_imputed)

normal_count = (predictions == 1).sum()
anomaly_count = (predictions == -1).sum()

total = len(predictions)
anomaly_percentage = (anomaly_count / total) * 100

print("\n[results]")
print(f"  Normal samples   : {normal_count}")
print(f"  Anomalous samples: {anomaly_count}")
print(f"  Anomaly percentage: {anomaly_percentage:.2f}%")


# ── Save model and imputer ───────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
joblib.dump(imputer, IMPUTER_PATH)

print("\n[save] Anomaly model saved:")
print(f"  {MODEL_PATH}")

print("[save] Anomaly imputer saved:")
print(f"  {IMPUTER_PATH}")

print("\n[done] Anomaly detection training complete.")