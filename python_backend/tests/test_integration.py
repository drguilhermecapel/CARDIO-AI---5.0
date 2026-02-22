import pytest
import numpy as np
from integration.cardio_ai_service import CardioAIIntegrationService

@pytest.fixture
def service():
    return CardioAIIntegrationService()

def test_full_pipeline_execution(service):
    # Mock Data
    ecg = np.random.randn(12, 5000)
    meta = {
        'id': 'TEST_PATIENT',
        'age': 55,
        'sex': 'Female',
        'symptoms': ['dyspnea'],
        'history': ['htn']
    }
    
    # Run pipeline
    result = service.process_ecg(ecg, meta)
    
    assert result['status'] == "COMPLETED"
    assert 'diagnosis' in result
    assert 'risk_profile' in result
    assert 'quality_check' in result

def test_pipeline_rejection_on_bad_quality(service):
    # Flatline signal (unusable)
    ecg = np.zeros((12, 5000))
    meta = {'id': 'BAD_ECG'}
    
    result = service.process_ecg(ecg, meta)
    
    assert result['status'] == "REJECTED"
    assert "unusable" in result['error']

def test_temporal_integration(service):
    ecg = np.random.randn(12, 5000)
    meta = {'id': 'TEMP_TEST'}
    prev = {'is_lbbb': False}
    
    # We can't easily force the diagnostic engine to output LBBB without mocking internal calls,
    # but we can check that the temporal analysis section exists in output
    result = service.process_ecg(ecg, meta, prev)
    
    assert 'temporal_analysis' in result
    assert result['temporal_analysis'] is not None
