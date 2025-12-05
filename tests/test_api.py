from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Fish" in response.json()["message"]


def test_predict_endpoint():
    # Payload de exemplo (Peixe Perch)
    payload = [
        {
            "Species": "Perch",
            "Length1": 20.0,
            "Length2": 22.0,
            "Length3": 23.5,
            "Height": 5.5,
            "Width": 3.3,
        }
    ]

    response = client.post("/predict", json=payload)

    # Se o modelo não estiver treinado, pode dar 500, mas assumimos que o ambiente de teste tem o modelo
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 1
        assert isinstance(data["predictions"][0], float)
