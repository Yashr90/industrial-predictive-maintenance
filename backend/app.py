from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "failure_model.joblib"
ANOMALY_PATH = ARTIFACTS / "anomaly_model.joblib"
METRICS_PATH = ARTIFACTS / "evaluation.json"
META_PATH = ARTIFACTS / "metadata.json"

app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description="Failure-risk classification and anomaly scoring demo using the UCI AI4I synthetic benchmark.",
    version="1.0.0"
)

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class SensorReading(BaseModel):
    machine_type: Literal["L", "M", "H"] = "M"
    air_temperature_k: float = Field(300.0, ge=250, le=400)
    process_temperature_k: float = Field(310.0, ge=250, le=450)
    rotational_speed_rpm: float = Field(1500, ge=0, le=10000)
    torque_nm: float = Field(40, ge=0, le=500)
    tool_wear_min: float = Field(120, ge=0, le=1000)

def load_artifact(path: Path):
    if not path.exists():
        raise RuntimeError(f"Model artifact missing: {path}. Run `python -m backend.train` first.")
    return joblib.load(path)

def input_frame(r: SensorReading) -> pd.DataFrame:
    return pd.DataFrame([{
        "Type": r.machine_type,
        "Air temperature [K]": r.air_temperature_k,
        "Process temperature [K]": r.process_temperature_k,
        "Rotational speed [rpm]": r.rotational_speed_rpm,
        "Torque [Nm]": r.torque_nm,
        "Tool wear [min]": r.tool_wear_min,
    }])

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists() and ANOMALY_PATH.exists()}

@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(503, "Evaluation metrics not found. Train the model first.")
    return json.loads(METRICS_PATH.read_text())

@app.post("/predict")
def predict(reading: SensorReading):
    try:
        classifier = load_artifact(MODEL_PATH)
        anomaly_bundle = load_artifact(ANOMALY_PATH)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    frame = input_frame(reading)
    probability = float(classifier.predict_proba(frame)[0, 1])
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {"threshold": 0.35}
    threshold = float(meta.get("threshold", 0.35))
    risk = "HIGH" if probability >= threshold else "LOW"

    transformed = anomaly_bundle["preprocess"].transform(frame)
    anomaly_model = anomaly_bundle["model"]
    anomaly_value = int(anomaly_model.predict(transformed)[0])  # -1 means unusual, +1 normal
    anomaly_score = float(-anomaly_model.score_samples(transformed)[0])

    return {
        "failure_probability": round(probability, 4),
        "risk_level": risk,
        "threshold": threshold,
        "anomaly_detected": anomaly_value == -1,
        "anomaly_score": round(anomaly_score, 4),
        "input": reading.model_dump(),
        "interpretation": (
            "Prioritize inspection based on this demo's risk score; validate against equipment-specific data."
            if risk == "HIGH" else
            "No high failure probability under this model's current threshold; continue routine monitoring."
        ),
        "limitations": "Benchmark model trained on synthetic data; not suitable for safety-critical decisions."
    }
