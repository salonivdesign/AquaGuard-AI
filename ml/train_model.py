"""
ml/train_model.py
-----------------
Day 2 -- Train and evaluate two classifiers for water potability prediction.

Models compared:
  1. Logistic Regression  (with StandardScaler, class_weight='balanced')
  2. Random Forest        (class_weight='balanced', best config chosen by CV)

Preprocessing follows Day 1 (ml/preprocess.py):
  - Median imputation fitted only on training data
  - Stratified 80/20 train-test split, random_state=42

RF hyperparameter selection uses 5-fold stratified cross-validation on the
training set only -- the test set is never touched during tuning.

The better model (selected on F1-score) plus all required preprocessing
components are saved to ml/ via joblib for use by predict.py.

Run:
    python ml/train_model.py
"""

import os
import sys
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# -- Reuse Day 1 preprocessing (no duplication) --------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import load_data, split_and_impute, FEATURES, TARGET  # noqa: E402

# -- Output paths --------------------------------------------------------------
ML_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(ML_DIR, "water_model.joblib")
IMPUTER_PATH = os.path.join(ML_DIR, "water_imputer.joblib")
SCALER_PATH  = os.path.join(ML_DIR, "water_scaler.joblib")

# -- RF configs to evaluate via cross-validation (training data only) ----------
# Each dict is passed directly to RandomForestClassifier(**cfg).
# Kept deliberately small -- this is not a grid search.
RF_CONFIGS = [
    dict(n_estimators=100, class_weight="balanced", random_state=42),
    dict(n_estimators=200, class_weight="balanced", random_state=42),
    dict(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42),
    dict(n_estimators=200, max_depth=15, min_samples_leaf=4,
         class_weight="balanced", random_state=42),
]


# -- Helpers -------------------------------------------------------------------

def _print_metrics(name, y_test, y_pred):
    """Print confusion matrix and return metrics dict."""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    metrics = {
        "name":      name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }

    print("  Confusion matrix -- " + name + ":")
    print("    TN=%4d  FP=%4d   (predicted Not-potable)" % (tn, fp))
    print("    FN=%4d  TP=%4d   (predicted Potable)" % (fn, tp))
    return metrics


def _print_comparison(results):
    sep = "-" * 68
    print("\n" + "=" * 68)
    print("  FINAL MODEL COMPARISON  (test set)")
    print("=" * 68)
    print("%-32s %9s %10s %9s %9s" % ("Model", "Accuracy", "Precision", "Recall", "F1"))
    print(sep)
    for r in results:
        print("  %-30s %9.4f %10.4f %9.4f %9.4f" % (
            r["name"], r["accuracy"], r["precision"], r["recall"], r["f1"]
        ))
    print(sep)


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 68)
    print("  AquaGuard -- Day 2: Model Training & Evaluation")
    print("=" * 68)

    # 1. Load & preprocess (Day 1 pipeline) ------------------------------------
    print("\n[step 1]  Loading data and applying Day 1 preprocessing ...")
    df = load_data()
    X_train_imp, X_test_imp, y_train, y_test = split_and_impute(df)

    # Refit imputer on the same training rows so we have a serialisable object.
    # Identical result is guaranteed by the same random_state + stratify=y.
    X = df[FEATURES]
    y = df[TARGET]
    X_train_raw, _, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_raw)

    # 2. Scale for Logistic Regression (fitted on training data only) ----------
    print("\n[step 2]  Fitting StandardScaler on training data (for LR) ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    # 3. Train Logistic Regression (class_weight='balanced') -------------------
    # Without class_weight, LR predicts all-0 because p(class=1) never
    # exceeds 0.5 with a 61/39 imbalance -- balancing corrects the threshold.
    print("\n[step 3]  Training Logistic Regression (class_weight='balanced') ...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    print("  [ok] Logistic Regression trained")

    # 4. Select best Random Forest config via 5-fold CV (train data only) ------
    print("\n[step 4]  Selecting Random Forest config via 5-fold stratified CV ...")
    print("          (test set is NOT touched at this stage)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_cv_f1   = -1.0
    best_rf_cfg  = None
    print()
    print("  %-60s  %s" % ("Configuration", "CV F1 (mean +/- std)"))
    print("  " + "-" * 72)
    for cfg in RF_CONFIGS:
        rf_candidate = RandomForestClassifier(**cfg)
        scores = cross_val_score(
            rf_candidate, X_train_imp, y_train, cv=cv, scoring="f1"
        )
        label = "n=%d d=%s l=%s" % (
            cfg["n_estimators"],
            cfg.get("max_depth", "None"),
            cfg.get("min_samples_leaf", 1),
        )
        print("  RF %-57s  %.4f +/- %.4f" % (label, scores.mean(), scores.std()))
        if scores.mean() > best_cv_f1:
            best_cv_f1  = scores.mean()
            best_rf_cfg = cfg

    print("\n  Best CV config: %s" % best_rf_cfg)
    print("  Best CV F1 mean: %.4f" % best_cv_f1)

    # 5. Train final RF with the best config on the full training set ----------
    print("\n[step 5]  Training final Random Forest with best config ...")
    rf = RandomForestClassifier(**best_rf_cfg)
    rf.fit(X_train_imp, y_train)
    print("  [ok] Random Forest trained")

    # 6. Evaluate both models on the held-out test set -------------------------
    print("\n[step 6]  Evaluating on test set ...")
    print()
    lr_metrics = _print_metrics("Logistic Regression (balanced)", y_test,
                                lr.predict(X_test_scaled))
    print()
    rf_metrics = _print_metrics("Random Forest (balanced, best CV)", y_test,
                                rf.predict(X_test_imp))

    results = [lr_metrics, rf_metrics]
    _print_comparison(results)

    # 7. Select the better model -----------------------------------------------
    print("\n[step 7]  Model selection ...")
    best_result  = max(results, key=lambda r: r["f1"])
    other_result = [r for r in results if r["name"] != best_result["name"]][0]

    if best_result["name"].startswith("Logistic"):
        best_model     = lr
        save_scaler    = scaler
        scaler_note    = "(required for Logistic Regression)"
    else:
        best_model     = rf
        save_scaler    = scaler    # always save so predict.py can load safely
        scaler_note    = "(saved but not applied -- RF is scale-invariant)"

    print("\n  Selected model : " + best_result["name"])
    print(
        "\n  Reason: %s achieves a higher F1-score (%.4f vs %.4f).\n"
        "  F1 is the primary selection criterion because the dataset has a\n"
        "  class imbalance (~61%% not-potable vs ~39%% potable); accuracy\n"
        "  alone would reward a model that predicts everything as class 0." % (
            best_result["name"], best_result["f1"], other_result["f1"]
        )
    )

    # 8. Save artifacts --------------------------------------------------------
    print("\n[step 8]  Saving artifacts ...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(imputer,    IMPUTER_PATH)
    joblib.dump(save_scaler, SCALER_PATH)

    print("  [saved] Model   -> " + os.path.relpath(MODEL_PATH))
    print("  [saved] Imputer -> " + os.path.relpath(IMPUTER_PATH))
    print("  [saved] Scaler  -> " + os.path.relpath(SCALER_PATH) + "  " + scaler_note)

    print("\n" + "=" * 68)
    print("  Training complete.")
    print("=" * 68)


if __name__ == "__main__":
    main()
