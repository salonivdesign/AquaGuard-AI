"""
ml/preprocess.py
----------------
Loads water_potability.csv, checks missing values, performs a stratified
80/20 train/test split, and applies median imputation fitted only on the
training set (no data leakage).

Run:
    python ml/preprocess.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "water_potability.csv")

TARGET = "Potability"
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines",
    "Sulfate", "Conductivity", "Organic_carbon",
    "Trihalomethanes", "Turbidity",
]
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ── Load ─────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  Shape: {df.shape}  |  Columns: {list(df.columns)}")
    return df


# ── Missing-value summary ─────────────────────────────────────────────────────
def missing_summary(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_%": pct})
    summary = summary[summary["missing_count"] > 0]
    if summary.empty:
        print("\n[missing]  No missing values found.")
    else:
        print("\n[missing]  Columns with missing values:")
        print(summary.to_string())


# ── Basic dataset statistics ──────────────────────────────────────────────────
def basic_stats(df: pd.DataFrame) -> None:
    print("\n[stats]  Descriptive statistics:")
    print(df[FEATURES].describe().round(4).to_string())


# ── Class distribution ────────────────────────────────────────────────────────
def class_distribution(y: pd.Series, label: str = "") -> None:
    counts = y.value_counts().sort_index()
    pcts = (y.value_counts(normalize=True).sort_index() * 100).round(2)
    print(f"\n[class dist{' ' + label if label else ''}]")
    for cls in counts.index:
        print(f"  Potability={cls}  ->  {counts[cls]:>5} rows  ({pcts[cls]:.2f}%)")


# ── Train/test split + median imputation ─────────────────────────────────────
def split_and_impute(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[FEATURES]
    y = df[TARGET]

    print("\n[split]  Stratified 80/20 split  (random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

    class_distribution(y_train, label="- train")
    class_distribution(y_test,  label="- test")

    # Median imputation - fit on train only
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    print("\n[impute]  Median imputation fitted on training set.")
    print("  Training medians:")
    for feat, median in zip(FEATURES, imputer.statistics_):
        print(f"    {feat:<20} : {median:.4f}")

    # Verify no remaining NaNs
    assert not np.isnan(X_train_imp).any(), "NaNs remain in X_train after imputation"
    assert not np.isnan(X_test_imp).any(),  "NaNs remain in X_test after imputation"
    print("\n[impute]  OK No missing values remain in train or test sets.")

    return X_train_imp, X_test_imp, y_train.values, y_test.values


# ── Feature scaling note ──────────────────────────────────────────────────────
def scaling_note() -> None:
    lines = [
        "",
        "[scaling]  Feature scaling guidance (for future model training):",
        "  * Logistic Regression  -> NEEDS scaling  (StandardScaler)",
        "  * SVM / SVR            -> NEEDS scaling  (StandardScaler)",
        "  * KNN                  -> NEEDS scaling  (StandardScaler / MinMaxScaler)",
        "  * Neural Networks      -> NEEDS scaling  (StandardScaler / MinMaxScaler)",
        "  * Random Forest        -> No scaling needed  (tree-based, scale-invariant)",
        "  * Gradient Boosting /",
        "    XGBoost / LightGBM   -> No scaling needed  (tree-based, scale-invariant)",
        "",
        "  Scalers should be fitted on X_train and applied to X_test (same leakage",
        "  rule as imputation). Scaling is NOT applied here; it belongs in each",
        "  model's own pipeline step.",
        "",
    ]
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    missing_summary(df)
    basic_stats(df)
    class_distribution(df[TARGET], label="- full dataset")
    X_train, X_test, y_train, y_test = split_and_impute(df)
    scaling_note()
    print("[done]  Preprocessing complete. Processed arrays are ready for ML pipeline.")
