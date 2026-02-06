import os
import numpy as np
import joblib
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

# --- Config ---
THRESHOLD = float(os.getenv("THRESHOLD", "0.088"))  # seuil métier
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "model.joblib")

# Optionnel: fallback MLflow (utile en local si tu veux)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/credit_scoring_xgboost/latest")

app = FastAPI(title="Credit Scoring API")

class PredictRequest(BaseModel):
    features: list[float]

# --- Load model ---
_model = None
_model_type = None  # "joblib" or "mlflow"

def load_model():
    global _model, _model_type
    if _model is not None:
        return

    # 1) Prefer local file (Render-friendly)
    if os.path.exists(LOCAL_MODEL_PATH):
        _model = joblib.load(LOCAL_MODEL_PATH)
        _model_type = "joblib"
        return

    # 2) Fallback: MLflow registry (local only unless you host MLflow)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model = mlflow.pyfunc.load_model(MODEL_URI)
    _model_type = "mlflow"

@app.on_event("startup")
def on_startup():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model_type": _model_type}

@app.post("/predict")
def predict(req: PredictRequest):
    load_model()
    X = np.array([req.features], dtype=float)

    if _model_type == "joblib":
        # XGBoost sklearn API
        p = float(_model.predict_proba(X)[:, 1][0])
    else:
        # MLflow pyfunc API
        proba = _model.predict(X)
        p = float(np.array(proba).reshape(-1)[0])

    decision = int(p >= THRESHOLD)  # 1=refus, 0=accepté
    return {"proba_default": p, "threshold": THRESHOLD, "decision": decision}
