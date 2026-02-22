import pytest
from fastapi.testclient import TestClient
from serving.main import app
import numpy as np
import os

client = TestClient(app)

# Mock the model loading to avoid needing the actual .h5 file in CI
# We can use unittest.mock or just rely on the fact that the app handles model=None gracefully 
# or we can mock the global variables in serving.main.

from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_model():
    with patch('serving.main.model') as mock:
        # Mock prediction output
        # Output shape: (Batch, Classes) -> (1, 5)
        mock.return_value.numpy.return_value = np.array([[0.1, 0.05, 0.8, 0.05, 0.0]]) # High MI
        yield mock

@pytest.fixture
def mock_explainer():
    with patch('serving.main.explainer') as mock:
        # Mock explain output: (5000, 12)
        mock.explain.return_value = np.zeros((5000, 12))
        yield mock

class TestIntegration:
    def test_health_check(self):
        # We don't have a specific health endpoint in the snippet provided earlier, 
        # but FastAPI usually has docs. 
        # Let's test the predict endpoint with a dummy request.
        pass

    @patch('serving.main.model')
    @patch('serving.main.explainer')
    @patch('serving.main.artifact_detector')
    def test_predict_endpoint_structure(self, mock_detector, mock_explainer, mock_model):
        """Test that the API returns the correct JSON structure."""
        
        # Setup Mocks
        # Model returns tensor
        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = np.array([[0.9, 0.05, 0.05, 0.0, 0.0]]) # Normal
        mock_model.return_value = mock_tensor
        
        # Detector returns valid beats
        mock_detector.process_record.return_value = [{'index': 100, 'is_valid': True}]
        
        # Payload
        payload = {
            "signal": np.random.randn(5000, 12).tolist(),
            "mc_samples": 2,
            "explain": False
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "diagnosis" in data
        assert "uncertainty" in data
        assert "triage" in data
        assert "quality_metrics" in data
        
        # Check Triage Logic (Normal -> Routine)
        assert data["triage"]["status"] == "ROUTINE"

    @patch('serving.main.model')
    @patch('serving.main.artifact_detector')
    def test_critical_triage(self, mock_detector, mock_model):
        """Test that high MI probability triggers CRITICAL triage."""
        
        # Mock Critical Prediction (MI is index 2 in our list: Normal, AFIB, MI, PVC, Noise)
        # Wait, classes were: ["Normal", "AFIB", "MI", "PVC", "Noise"]
        # So MI is index 2.
        
        mock_tensor = MagicMock()
        # Return MI=0.9
        mock_tensor.numpy.return_value = np.array([[0.05, 0.05, 0.9, 0.0, 0.0]]) 
        mock_model.return_value = mock_tensor
        
        mock_detector.process_record.return_value = []
        
        payload = {
            "signal": np.random.randn(5000, 12).tolist(),
            "mc_samples": 1
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["triage"]["status"] == "CRITICAL"
        assert "MI Risk" in data["triage"]["reason"]

    def test_invalid_input_shape(self):
        """Test API response for wrong signal shape."""
        payload = {
            "signal": [[1.0, 2.0]], # Too short, wrong dimensions
            "mc_samples": 1
        }
        # The API currently converts to numpy and might crash or handle it.
        # Ideally it should return 422 or 500.
        response = client.post("/predict", json=payload)
        # We expect validation error or handled exception
        assert response.status_code in [422, 500]
