import os
import numpy as np
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

# --- Config ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/credit_scoring_xgboost/latest")
THRESHOLD = float(os.getenv("THRESHOLD", "0.088"))  # mets ton seuil métier

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Credit Scoring API")

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array([req.features], dtype=float)
    proba = model.predict(X)

    # model.predict peut renvoyer [p] ou [[p]] selon les cas
    p = float(np.array(proba).reshape(-1)[0])

    decision = int(p >= THRESHOLD)  # 1=refus, 0=accepté (à expliciter dans ton doc)
    return {"proba_default": p, "threshold": THRESHOLD, "decision": decision}
