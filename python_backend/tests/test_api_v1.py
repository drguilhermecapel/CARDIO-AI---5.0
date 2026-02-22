import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_analyze_ecg_unauthorized():
    # No token
    response = client.post("/api/v1/ecg/analyze", json={
        "patient_id": "123",
        "raw_data": [0.1, 0.2, 0.3],
        "device_id": "dev_1"
    })
    assert response.status_code == 401

# Note: Full integration tests require mocking DB and Auth
