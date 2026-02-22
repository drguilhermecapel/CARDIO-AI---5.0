import pytest
import numpy as np
from signal_processor_v2 import ECGSignalProcessor

@pytest.fixture
def processor():
    return ECGSignalProcessor(sampling_rate=500)

@pytest.fixture
def normal_ecg():
    # 12 leads, 5 seconds (2500 samples)
    t = np.linspace(0, 5, 2500)
    sig = np.sin(2 * np.pi * 1.0 * t) # 1Hz sine wave as dummy ECG
    return np.tile(sig, (12, 1))

def test_normal_ecg_processing(processor, normal_ecg):
    result = processor.process(normal_ecg)
    assert result['quality_score'] == 100.0
    assert result['artifacts']['line_noise'] is False
    assert len(result['features']) >= 40

def test_60hz_noise_detection(processor, normal_ecg):
    t = np.linspace(0, 5, 2500)
    noise = 2.0 * np.sin(2 * np.pi * 60.0 * t) # Strong 60Hz
    noisy_ecg = normal_ecg + noise
    
    # Detect artifacts directly
    artifacts = processor.detect_artifacts(noisy_ecg)
    assert artifacts['line_noise'] is True
    
    # Process and check score penalty
    result = processor.process(noisy_ecg)
    assert result['quality_score'] < 100

def test_motion_artifact_detection(processor, normal_ecg):
    # Add burst of noise
    noisy_ecg = normal_ecg.copy()
    # Add high amplitude noise to Lead V1 (index 6) in the middle
    noisy_ecg[6, 1000:1500] += np.random.normal(0, 5.0, 500) 
    
    artifacts = processor.detect_artifacts(noisy_ecg)
    assert artifacts['motion_artifact'] is True
    assert 6 in artifacts['leads_with_noise']

def test_lead_disconnection(processor, normal_ecg):
    # Flatline Lead I (index 0)
    disconnected_ecg = normal_ecg.copy()
    disconnected_ecg[0, :] = 0.0 # Flatline
    
    artifacts = processor.detect_artifacts(disconnected_ecg)
    assert 0 in artifacts['lead_disconnection']
    
    result = processor.process(disconnected_ecg)
    assert result['quality_score'] <= 90 # Penalty applied
