from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Credit Scoring API", version="1.0.0")

# =========================
# Paramètres
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "model.joblib"
DATA_PATH = BASE_DIR / "data" / "test_clients.csv"

ID_COL = "SK_ID_CURR"
THRESHOLD = 0.5  # tu pourras le remplacer plus tard par ton vrai seuil métier

# =========================
# Chargement au démarrage
# =========================
model = joblib.load(MODEL_PATH)
df_clients = pd.read_csv(DATA_PATH)


def convert_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes object en numérique quand c'est possible.
    Les valeurs non convertibles deviennent NaN.
    """
    df = df.copy()

    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@app.get("/")
def root():
    return {"message": "API de scoring active"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/clients")
def get_clients():
    return {
        "n_clients": int(df_clients.shape[0]),
        "client_ids": df_clients[ID_COL].tolist()
    }


@app.get("/predict/{client_id}")
def predict_client(client_id: int):
    client_row = df_clients[df_clients[ID_COL] == client_id].copy()

    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client introuvable")

    # Important : on garde l'ID car le modèle actuel l'attend
    X_client = convert_object_columns(client_row)

    try:
        proba = float(model.predict_proba(X_client)[:, 1][0])
        pred = int(proba >= THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

    return {
        "client_id": int(client_id),
        "default_probability": round(proba, 6),
        "prediction": pred,
        "threshold": THRESHOLD
    }