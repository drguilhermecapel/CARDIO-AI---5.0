import pytest
import numpy as np
from signal_processing import ECGValidator, ECGPreprocessor, ArtifactDetector, process_ecg_pipeline

@pytest.fixture
def synthetic_ecg():
    # Generate 12 leads, 10 seconds, 500Hz
    fs = 500
    t = np.linspace(0, 10, 10 * fs)
    # Simple synthetic ECG: Sine wave + Noise
    ecg = np.sin(2 * np.pi * 1.0 * t) # 1Hz heart rate
    noise = np.random.normal(0, 0.1, len(t))
    signal = ecg + noise
    # Replicate to 12 leads
    return np.tile(signal, (12, 1))

def test_validator_valid(synthetic_ecg):
    validator = ECGValidator()
    valid, msg = validator.validate(synthetic_ecg, 500)
    assert valid is True
    assert msg == "Valid"

def test_validator_short_duration():
    validator = ECGValidator(min_duration_sec=8.0)
    short_sig = np.zeros((12, 500 * 5)) # 5 seconds
    valid, msg = validator.validate(short_sig, 500)
    assert valid is False
    assert "Duration" in msg

def test_validator_flatline():
    validator = ECGValidator()
    flat_sig = np.zeros((12, 500 * 10))
    valid, msg = validator.validate(flat_sig, 500)
    assert valid is False
    assert "Flatline" in msg

def test_preprocessor_shape(synthetic_ecg):
    pre = ECGPreprocessor(fs=500)
    processed = pre.process(synthetic_ecg)
    assert processed.shape == synthetic_ecg.shape
    # Check normalization (0-1)
    assert np.max(processed) <= 1.0 + 1e-6
    assert np.min(processed) >= 0.0 - 1e-6

def test_artifact_detector_score(synthetic_ecg):
    det = ArtifactDetector(fs=500)
    metrics = det.check_artifacts(synthetic_ecg)
    assert 'quality_score' in metrics
    assert 0 <= metrics['quality_score'] <= 100
    assert metrics['status'] in ['ACCEPTED', 'REJECTED']

def test_pipeline_performance(synthetic_ecg):
    import time
    start = time.time()
    result = process_ecg_pipeline(synthetic_ecg, fs=500)
    duration = time.time() - start
    assert duration < 1.0 # Requirement: < 1s
    assert result['status'] == 'SUCCESS'
