# AquaGuard AI

## AI-Based Water Quality Risk Assessment and Prevention Support System

AquaGuard AI is a machine-learning-based prototype designed to assess
water-quality risk from measured water-quality parameters and identify
statistically unusual patterns.

The system is designed as a decision-support tool for environments such
as hostels, households, schools, and small institutions. It provides
risk assessment and preventive recommendations based on the available
water-quality data.

> AquaGuard AI is a prototype decision-support system. It does not
> replace laboratory water testing or certify water as safe for drinking.

---

## Problem Statement

Water quality can deteriorate due to changes in different physical and
chemical parameters. In many settings, users may not have an easy way
to interpret multiple water-quality measurements together.

The problem addressed by AquaGuard AI is:

**How might we use AI to identify water-quality risks and unusual
measurement patterns and help users take preventive action before a
potential water-quality problem becomes serious?**

---

## Objectives

- Assess water-quality/potability risk using machine learning.
- Detect statistically unusual combinations of water-quality parameters.
- Provide an overall risk level.
- Provide preventive recommendations based on the detected risk.
- Maintain a history of previous analyses.
- Provide a simple web-based interface for users.

---

## AI Approach

AquaGuard AI uses two machine-learning approaches:

### 1. Random Forest Classification

A Random Forest classifier analyzes nine water-quality parameters and
estimates whether the input resembles potable or non-potable samples
based on patterns learned from the training dataset.

### 2. Isolation Forest Anomaly Detection

An Isolation Forest identifies statistically unusual combinations of
water-quality parameters.

An anomaly indicates an unusual statistical pattern. It does not by
itself prove contamination.

### 3. Combined Risk Assessment

The outputs of both models are combined to produce an overall risk level:

- LOW
- MEDIUM
- HIGH

The system then provides a corresponding recommendation to support
preventive action.

---

## Water Quality Parameters

The system uses the following parameters:

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

---

## Technology Stack

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Backend
- FastAPI
- Uvicorn

### Database
- SQLite

### Frontend
- HTML
- CSS
- JavaScript

### Development
- Git
- GitHub
- VS Code

---

## System Workflow

```text
Water Quality Parameters
          ↓
    Data Preprocessing
          ↓
 ┌────────┴─────────┐
 ↓                  ↓
Random Forest   Isolation Forest
 ↓                  ↓
Potability       Anomaly
Risk             Detection
 └────────┬─────────┘
          ↓
    Overall Risk
          ↓
   Recommendation
          ↓
   SQLite History
          ↓
    Web Dashboard

---

## Sustainable Development Goal Alignment

AquaGuard AI aligns primarily with:

### SDG 6 — Clean Water and Sanitation

The project supports SDG 6 by promoting better awareness and monitoring
of drinking-water quality. By analyzing water-quality parameters and
highlighting potential risks, AquaGuard AI can help users identify when
additional testing or preventive action may be appropriate.

The system supports responsible water-quality management but does not
replace professional water testing or water-treatment systems.

### SDG 3 — Good Health and Well-Being

Safe drinking water is important for protecting public health. AquaGuard
AI supports preventive awareness by helping users interpret water-quality
measurements and identify potential concerns that may require further
investigation.

---

## Expected Impact

AquaGuard AI is intended to:

- Improve awareness of drinking-water quality.
- Help users interpret multiple water-quality measurements together.
- Encourage earlier preventive action when potential risks are detected.
- Maintain a record of previous water-quality analyses.
- Support better water-quality monitoring in hostels, households, and
  small institutions.

The prototype demonstrates how AI can be used as a decision-support tool
for sustainability and preventive water-quality management.