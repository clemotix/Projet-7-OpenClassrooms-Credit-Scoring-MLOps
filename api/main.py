import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException

# --- Config ---
THRESHOLD = float(os.getenv("THRESHOLD", "0.515"))
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "api/model_xgb_final.joblib")
CLIENTS_DATA_PATH = os.getenv("CLIENTS_DATA_PATH", "api/data/clients_render.csv")

app = FastAPI(title="Credit Scoring API")

_model = None
_clients_df = None


def load_model():
    global _model
    if _model is not None:
        return

    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {LOCAL_MODEL_PATH}")

    _model = joblib.load(LOCAL_MODEL_PATH)


def load_clients_data():
    global _clients_df
    if _clients_df is not None:
        return

    if not os.path.exists(CLIENTS_DATA_PATH):
        raise FileNotFoundError(f"Fichier introuvable : {CLIENTS_DATA_PATH}")

    _clients_df = pd.read_csv(CLIENTS_DATA_PATH)


@app.on_event("startup")
def on_startup():
    load_model()
    load_clients_data()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": "joblib",
        "n_clients": len(_clients_df) if _clients_df is not None else 0
    }


@app.get("/clients")
def get_clients():
    load_clients_data()
    return {
        "n_clients": len(_clients_df),
        "client_ids": _clients_df["SK_ID_CURR"].tolist()
    }


@app.get("/predict/{client_id}")
def predict(client_id: int):
    load_model()
    load_clients_data()

    client_row = _clients_df[_clients_df["SK_ID_CURR"] == client_id]

    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client introuvable")

    # Important :
    # ton modèle actuel attend encore SK_ID_CURR dans les features,
    # donc on garde toutes les colonnes telles quelles
    X = client_row.astype(float)

    p = float(_model.predict_proba(X)[:, 1][0])
    decision = int(p >= THRESHOLD)

    return {
        "client_id": client_id,
        "default_probability": round(p, 6),
        "threshold": THRESHOLD,
        "decision": decision
    }