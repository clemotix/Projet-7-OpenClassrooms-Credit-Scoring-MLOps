import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# --- Config ---
THRESHOLD = float(os.getenv("THRESHOLD", "0.515"))
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "api/model_xgb_final.joblib")
CLIENTS_DATA_PATH = os.getenv("CLIENTS_DATA_PATH", "api/data/clients_render.csv")

app = FastAPI(title="Credit Scoring API")

_model = None
_clients_df = None


# ---------------------------
# Schéma de simulation
# ---------------------------
class SimulationInput(BaseModel):
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    CNT_CHILDREN: Optional[int] = None
    CNT_FAM_MEMBERS: Optional[float] = None


# ---------------------------
# Chargements
# ---------------------------
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


# ---------------------------
# Fonctions utilitaires
# ---------------------------
def recompute_derived_features(row: pd.Series) -> pd.Series:
    """
    Recalcule les variables dérivées impactées par la simulation.
    """
    income = row.get("AMT_INCOME_TOTAL", np.nan)
    credit = row.get("AMT_CREDIT", np.nan)
    annuity = row.get("AMT_ANNUITY", np.nan)
    fam_members = row.get("CNT_FAM_MEMBERS", np.nan)
    days_employed = row.get("DAYS_EMPLOYED", np.nan)

    if pd.notna(days_employed) and pd.notna(row.get("DAYS_BIRTH", np.nan)) and row["DAYS_BIRTH"] != 0:
        row["DAYS_EMPLOYED_PERC"] = days_employed / row["DAYS_BIRTH"]
    else:
        row["DAYS_EMPLOYED_PERC"] = np.nan

    if pd.notna(income) and income != 0:
        row["INCOME_CREDIT_PERC"] = credit / income if pd.notna(credit) else np.nan
        row["ANNUITY_INCOME_PERC"] = annuity / income if pd.notna(annuity) else np.nan
    else:
        row["INCOME_CREDIT_PERC"] = np.nan
        row["ANNUITY_INCOME_PERC"] = np.nan

    if pd.notna(fam_members) and fam_members != 0 and pd.notna(income):
        row["INCOME_PER_PERSON"] = income / fam_members
    else:
        row["INCOME_PER_PERSON"] = np.nan

    if pd.notna(credit) and credit != 0 and pd.notna(annuity):
        row["PAYMENT_RATE"] = annuity / credit
    else:
        row["PAYMENT_RATE"] = np.nan

    return row


def score_row(row: pd.Series) -> tuple[float, int]:
    X = pd.DataFrame([row]).astype(float)
    p = float(_model.predict_proba(X)[:, 1][0])
    decision = int(p >= THRESHOLD)
    return p, decision


# ---------------------------
# Routes
# ---------------------------
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

    row = client_row.iloc[0].copy()
    p, decision = score_row(row)

    return {
        "client_id": client_id,
        "default_probability": round(p, 6),
        "threshold": THRESHOLD,
        "decision": decision
    }


@app.post("/simulate/{client_id}")
def simulate(client_id: int, inputs: SimulationInput):
    load_model()
    load_clients_data()

    client_row = _clients_df[_clients_df["SK_ID_CURR"] == client_id]

    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client introuvable")

    original_row = client_row.iloc[0].copy()
    simulated_row = original_row.copy()

    updates = inputs.model_dump(exclude_none=True)

    if not updates:
        raise HTTPException(status_code=400, detail="Aucune variable modifiée fournie")

    for col, value in updates.items():
        if col not in simulated_row.index:
            raise HTTPException(status_code=400, detail=f"Colonne inconnue : {col}")
        simulated_row[col] = value

    simulated_row = recompute_derived_features(simulated_row)

    original_p, original_decision = score_row(original_row)
    simulated_p, simulated_decision = score_row(simulated_row)

    return {
        "client_id": client_id,
        "threshold": THRESHOLD,
        "original": {
            "default_probability": round(original_p, 6),
            "decision": original_decision
        },
        "simulated": {
            "default_probability": round(simulated_p, 6),
            "decision": simulated_decision
        },
        "updated_fields": updates
    }