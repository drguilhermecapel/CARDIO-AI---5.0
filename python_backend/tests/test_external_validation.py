import pytest
import os
import json
from validation.external_validator import ExternalValidator
from validation.data_loaders import MockPTBXLLoader
from integration.cardio_ai_service import CardioAIIntegrationService

def test_external_validation_flow(tmpdir):
    # Setup
    service = CardioAIIntegrationService()
    validator = ExternalValidator(service)
    loader = MockPTBXLLoader(n_samples=10)
    
    # Run
    validator.run_validation(loader, "Test_Dataset")
    
    # Check Output
    assert os.path.exists("validation_results_Test_Dataset.json")
    
    with open("validation_results_Test_Dataset.json", "r") as f:
        metrics = json.load(f)
        
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert metrics["total_cases"] == 10
    
    # Clean up
    os.remove("validation_results_Test_Dataset.json")
