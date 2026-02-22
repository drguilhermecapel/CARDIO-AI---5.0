import pytest
import numpy as np
from calibration.probability_calibrator import ProbabilityCalibrator

@pytest.fixture
def calibrator():
    return ProbabilityCalibrator(method='isotonic')

def test_isotonic_calibration(calibrator):
    # Perfectly calibrated input
    probs = np.linspace(0, 1, 100)
    labels = np.random.binomial(1, probs)
    
    calibrator.fit(probs, labels)
    cal_probs = calibrator.calibrate(probs)
    
    # Should remain close to input
    assert np.mean(np.abs(cal_probs - probs)) < 0.15

def test_temperature_scaling():
    calibrator = ProbabilityCalibrator(method='temperature_scaling')
    
    # Logits
    logits = np.random.randn(100, 3)
    labels = np.random.randint(0, 3, 100)
    labels_onehot = np.eye(3)[labels]
    
    calibrator.fit(np.zeros((100, 3)), labels_onehot, logits=logits)
    
    assert calibrator.is_fitted
    assert calibrator.temperature > 0
    
    cal_probs = calibrator.calibrate(None, logits=logits)
    assert cal_probs.shape == (100, 3)
    assert np.allclose(np.sum(cal_probs, axis=1), 1.0)

def test_reliability_metrics(calibrator):
    probs = np.array([0.1, 0.9, 0.8, 0.2])
    labels = np.array([0, 1, 1, 0])
    
    metrics = calibrator.assess_reliability(labels, probs, n_bins=2, n_bootstrap=10)
    
    assert 'ECE' in metrics
    assert 'MCE' in metrics
    assert 'Brier' in metrics
    assert 'Curve' in metrics
    assert len(metrics['Curve']['prob_pred']) > 0

def test_overconfidence_detection(calibrator):
    # Predict 0.9 everywhere, but accuracy is 0.5
    probs = np.ones(100) * 0.9
    labels = np.concatenate([np.ones(50), np.zeros(50)]) # 0.5 acc
    
    metrics = calibrator.assess_reliability(labels, probs, n_bins=1)
    
    assert metrics['Bias'] > 0.3 # 0.9 - 0.5 = 0.4
    assert metrics['Status'] == "Overconfident"
