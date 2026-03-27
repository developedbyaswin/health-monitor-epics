
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Poshan Mithra ML API",
    description="Nutrition risk prediction for rural Madhya Pradesh",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─── Load model at startup ──────────────────────────────────
model = None

@app.on_event("startup")
def load_model():
    global model
    paths = ["poshan_model.pkl", "/data/poshan_model.pkl"]
    for p in paths:
        if os.path.exists(p):
            model = joblib.load(p)
            logger.info(f"✅ Model loaded from {p}")
            return
    logger.error("❌ Model file not found!")
    raise FileNotFoundError("poshan_model.pkl not found")



class AssessmentInput(BaseModel):
    # Direct dataset fields
    age:                int   = Field(..., example=28)
    gender:             int   = Field(..., example=0,   description="0=female 1=male")
    height_cm:          float = Field(..., example=158)
    weight_kg:          float = Field(..., example=45)
    meals_per_day:      int   = Field(..., example=2,   description="1–4")
    diet_type:          int   = Field(..., example=0,   description="0=non-veg 1=veg")
    water_lpd:          float = Field(..., example=1.5, description="Litres per day")
    physical_activity:  int   = Field(..., example=1,   description="0=none 1=low 2=mod 3=high")
    chronic_disease:    int   = Field(..., example=0,   description="0=no 1=yes")
    appetite_loss:      int   = Field(..., example=0,   description="0=no 1=yes")
    fatigue:            int   = Field(..., example=1,   description="0=no 1=yes")
    pregnant_lactating: int   = Field(..., example=0,   description="0=no 1=yes")

class PredictionResponse(BaseModel):
    risk_level:      str    # "low" | "moderate" | "high"
    risk_score:      int    # 0–100
    confidence:      float  # 0.0–1.0
    probabilities:   dict   # {low, moderate, high}
    bmi:             float
    bmi_category:    str
    used_ml_model:   bool = True
    model_version:   str  = "gradboost-v2"


LABEL_MAP = {0: "low", 1: "moderate", 2: "high"}

def bmi_category_str(bmi: float) -> str:
    if bmi < 16:   return "Severely Underweight"
    if bmi < 18.5: return "Underweight"
    if bmi < 25:   return "Normal"
    if bmi < 30:   return "Overweight"
    return "Obese"

def encode_features(data: AssessmentInput):
    bmi = data.weight_kg / ((data.height_cm / 100) ** 2)

    def bmi_cat(b):
        if b < 16:   return 0
        if b < 18.5: return 1
        if b < 25:   return 2
        if b < 30:   return 3
        return 4

    w = data.water_lpd
    water_cat = 0 if w <= 1.0 else 1 if w <= 1.5 else 2 if w <= 2.0 else 3

    a = data.age
    age_grp = 0 if a <= 18 else 1 if a <= 30 else 2 if a <= 50 else 3

    sym_count = (data.appetite_loss + data.fatigue +
                 data.chronic_disease + data.pregnant_lactating)

    diet_activity       = data.diet_type * data.physical_activity
    nutrition_risk_flag = (
        (1 if data.meals_per_day <= 2 else 0) +
        (1 if data.water_lpd < 1.5 else 0) +
        (1 if data.diet_type == 0 else 0)
    )

    features = [
        
        data.age, data.gender, data.height_cm, data.weight_kg, round(bmi, 2),
        
        data.meals_per_day, data.diet_type, data.water_lpd, data.physical_activity,
        
        data.chronic_disease, data.appetite_loss, data.fatigue, data.pregnant_lactating,
        
        bmi_cat(bmi), sym_count, water_cat,
        age_grp, diet_activity, nutrition_risk_flag,
    ]

    return np.array(features, dtype=np.float32).reshape(1, -1), round(bmi, 2)


@app.post("/predict", response_model=PredictionResponse)
def predict(data: AssessmentInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features, bmi = encode_features(data)
        raw   = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        risk_level = LABEL_MAP.get(int(raw), "moderate")
        confidence = float(max(proba))

        
        base  = {"low": 10, "moderate": 50, "high": 90}[risk_level]
        score = int(base + (confidence - 0.5) * 30)
        score = max(0, min(100, score))

        proba_dict = {
            "low":      round(float(proba[0]), 3),
            "moderate": round(float(proba[1]), 3),
            "high":     round(float(proba[2]), 3),
        }

        logger.info(f"→ {risk_level} (conf={confidence:.2f} bmi={bmi})")
        return PredictionResponse(
            risk_level=risk_level, risk_score=score,
            confidence=round(confidence, 4), probabilities=proba_dict,
            bmi=bmi, bmi_category=bmi_category_str(bmi),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok" if model else "model_missing", "model": "GradientBoosting v2"}

@app.get("/")
def root():
    return {"message": "Poshan Mithra ML API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
