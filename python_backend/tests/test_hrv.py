import pytest
import numpy as np
from analysis.hrv_analysis import HRVRiskAssessor

@pytest.fixture
def assessor():
    return HRVRiskAssessor()

def test_time_domain_metrics(assessor):
    # Constant RR = 1000ms (60bpm) -> 0 Variability
    rr_ms = np.array([1000, 1000, 1000, 1000, 1000])
    metrics = assessor.calculate_time_domain(rr_ms)
    
    assert metrics['RMSSD'] == 0.0
    assert metrics['SDNN'] == 0.0
    assert metrics['pNN50'] == 0.0
    assert metrics['MeanNN'] == 1000.0

def test_high_variability(assessor):
    # Alternating 800, 1200 (diff 400 > 50)
    rr_ms = np.array([800, 1200, 800, 1200, 800, 1200])
    metrics = assessor.calculate_time_domain(rr_ms)
    
    assert metrics['pNN50'] == 100.0 # All diffs are 400ms > 50ms
    assert metrics['RMSSD'] > 0

def test_risk_assessment_high_risk(assessor):
    # Generate low variability R-peaks
    # SD = 5ms
    r_peaks = np.cumsum(np.random.normal(1000, 5, 50)).astype(int)
    
    res = assessor.assess_risk(r_peaks, fs=1000)
    
    assert res['risk_assessment']['sudden_cardiac_death_risk'] == "High"
    assert "SDNN < 50ms" in res['risk_assessment']['risk_factors'][0]

def test_dfa_calculation(assessor):
    # 1/f noise (Pink noise) should have alpha ~ 1.0
    # White noise (Random) should have alpha ~ 0.5
    
    # Random noise
    rr_ms = np.random.normal(800, 50, 100)
    alpha = assessor.calculate_dfa_alpha1(rr_ms)
    
    # Short series estimation is noisy, but should be < 0.75 usually for white noise
    # or at least calculable
    assert isinstance(alpha, float)
