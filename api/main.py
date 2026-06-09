"""
FastAPI Inference Backend — CVD Risk Predictor
==============================================
Serves predictions from the calibrated XGBoost model.
SHAP values are computed at inference time for explainability.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

def _load(filename):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path} — run training notebooks first.")
    return joblib.load(path)

def _load_json(filename):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path) as f:
        return json.load(f)

try:
    CALIBRATOR   = _load("calibrator.pkl")
    XGB_MODEL    = _load("xgboost_model.pkl")
    SCALER       = _load("scaler.pkl")
    FEATURE_COLS = _load_json("feature_columns.json")
    EXPLAINER    = shap.TreeExplainer(XGB_MODEL)
    logger.info("All artifacts loaded. Features: %s", FEATURE_COLS)
except FileNotFoundError as e:
    logger.error(str(e))
    CALIBRATOR = XGB_MODEL = SCALER = FEATURE_COLS = EXPLAINER = None

# ---------------------------------------------------------------------------
# LRS computation
# ---------------------------------------------------------------------------

LRS_COMPONENTS = ["sleep_regularity", "activity_mets", "smoking_packyears",
                  "sedentary_hours", "alcohol_units"]
LRS_WEIGHTS    = [0.2, 0.2, 0.2, 0.2, 0.2]
_NORM_BOUNDS   = {
    "sleep_regularity":  {"min": 0.2,  "max": 4.0,  "invert": False},
    "activity_mets":     {"min": 5.0,  "max": 60.0, "invert": True},
    "smoking_packyears": {"min": 0.0,  "max": 50.0, "invert": False},
    "sedentary_hours":   {"min": 2.0,  "max": 16.0, "invert": False},
    "alcohol_units":     {"min": 0.0,  "max": 40.0, "invert": False},
}

def compute_lrs_single(sleep_regularity, activity_mets, smoking_packyears,
                       sedentary_hours, alcohol_units) -> float:
    values = {
        "sleep_regularity":  sleep_regularity,
        "activity_mets":     activity_mets,
        "smoking_packyears": smoking_packyears,
        "sedentary_hours":   sedentary_hours,
        "alcohol_units":     alcohol_units,
    }
    lrs = 0.0
    for component, weight in zip(LRS_COMPONENTS, LRS_WEIGHTS):
        v      = values[component]
        bounds = _NORM_BOUNDS[component]
        lo, hi = bounds["min"], bounds["max"]
        v      = max(lo, min(hi, v))
        norm   = (v - lo) / (hi - lo) if hi > lo else 0.0
        if bounds["invert"]:
            norm = 1.0 - norm
        lrs += weight * norm
    return round(float(lrs), 6)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    age:                 float = Field(..., ge=18,  le=120)
    sex:                 int   = Field(..., ge=0,   le=1)
    cholesterol:         float = Field(..., ge=100, le=600)
    systolic_bp:         float = Field(..., ge=70,  le=250)
    fasting_blood_sugar: int   = Field(..., ge=0,   le=1)
    max_heart_rate:      float = Field(..., ge=40,  le=220)
    sleep_regularity:    float = Field(..., ge=0.2, le=4.0)
    activity_mets:       float = Field(..., ge=5.0, le=60.0)
    smoking_packyears:   float = Field(..., ge=0.0, le=50.0)
    sedentary_hours:     float = Field(..., ge=2.0, le=16.0)
    alcohol_units:       float = Field(..., ge=0.0, le=40.0)

    @field_validator("sex", "fasting_blood_sugar")
    @classmethod
    def must_be_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("Must be 0 or 1")
        return v


class RiskBand(BaseModel):
    label:       str
    color:       str
    description: str


class SHAPContribution(BaseModel):
    feature:    str
    value:      float
    shap_value: float
    direction:  str


class PredictResponse(BaseModel):
    risk_probability:   float
    risk_percent:       str
    lrs:                float
    lrs_percent:        str
    risk_band:          RiskBand
    shap_contributions: list
    expected_value:     float
    model_used:         str

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="CVD Risk Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status":       "ok" if CALIBRATOR is not None else "degraded",
        "model_loaded": CALIBRATOR is not None,
        "features":     FEATURE_COLS,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if CALIBRATOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    lrs = compute_lrs_single(
        req.sleep_regularity, req.activity_mets, req.smoking_packyears,
        req.sedentary_hours,  req.alcohol_units,
    )

    raw = {
        "age":                 req.age,
        "sex":                 req.sex,
        "cholesterol":         req.cholesterol,
        "systolic_bp":         req.systolic_bp,
        "fasting_blood_sugar": req.fasting_blood_sugar,
        "max_heart_rate":      req.max_heart_rate,
        "LRS":                 lrs,
    }

    missing = [c for c in FEATURE_COLS if c not in raw]
    if missing:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: {missing}")

    input_df     = pd.DataFrame([{c: raw[c] for c in FEATURE_COLS}])
    input_scaled = pd.DataFrame(SCALER.transform(input_df), columns=FEATURE_COLS)

    prob = float(CALIBRATOR.predict_proba(input_scaled)[0, 1])

    # SHAP from base XGBoost (TreeExplainer needs tree access)
    shap_vals = EXPLAINER.shap_values(input_scaled)
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[1][0]
    else:
        shap_arr = shap_vals[0]

    exp_val = EXPLAINER.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        exp_val = float(exp_val[1])
    else:
        exp_val = float(exp_val)

    contributions = []
    for i, col in enumerate(FEATURE_COLS):
        sv = float(shap_arr[i])
        contributions.append({
            "feature":    col,
            "value":      round(float(raw[col]), 4),
            "shap_value": round(sv, 6),
            "direction":  "increases_risk" if sv > 0.005
                          else "decreases_risk" if sv < -0.005
                          else "neutral",
        })
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    if prob < 0.10:
        band = {"label": "Low",       "color": "#22c55e",
                "description": "Low 10-year CVD risk. Maintain healthy lifestyle."}
    elif prob < 0.20:
        band = {"label": "Moderate",  "color": "#f59e0b",
                "description": "Moderate risk. Consider lifestyle modifications."}
    elif prob < 0.30:
        band = {"label": "High",      "color": "#f97316",
                "description": "High risk. Consult a cardiologist."}
    else:
        band = {"label": "Very High", "color": "#ef4444",
                "description": "Very high risk. Immediate medical consultation advised."}

    return {
        "risk_probability":   round(prob, 6),
        "risk_percent":       f"{prob * 100:.1f}%",
        "lrs":                lrs,
        "lrs_percent":        f"{lrs * 100:.1f}%",
        "risk_band":          band,
        "shap_contributions": contributions,
        "expected_value":     round(exp_val, 6),
        "model_used":         "XGBoost + Isotonic Calibration",
    }