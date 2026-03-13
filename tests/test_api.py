from api.app import app, df_clients, ID_COL
from fastapi.testclient import TestClient

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de scoring active"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_clients():
    response = client.get("/clients")
    assert response.status_code == 200

    data = response.json()

    assert "n_clients" in data
    assert "client_ids" in data
    assert data["n_clients"] == 100
    assert len(data["client_ids"]) == 100


def test_predict_existing_client():
    existing_client_id = int(df_clients[ID_COL].iloc[0])

    response = client.get(f"/predict/{existing_client_id}")
    assert response.status_code == 200

    data = response.json()

    assert data["client_id"] == existing_client_id
    assert "default_probability" in data
    assert "prediction" in data
    assert "threshold" in data
    assert data["prediction"] in [0, 1]


def test_predict_unknown_client():
    response = client.get("/predict/999999999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Client introuvable"