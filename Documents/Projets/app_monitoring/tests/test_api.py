from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_success():
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=test_data)

    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_data():
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post("/predict", json=test_data)

    assert response.status_code == 422