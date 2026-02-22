import pytest
import numpy as np
from quality.quality_assessor import QualityAssessor

@pytest.fixture
def assessor():
    return QualityAssessor(fs=500)

def test_clean_signal(assessor):
    t = np.linspace(0, 5, 2500)
    sig = np.tile(np.sin(2*np.pi*1*t), (12, 1))
    report = assessor.assess_quality(sig)
    
    assert report['overall_score'] == 100.0
    assert report['reliability'] == "High"
    assert len(report['issues_detected']) == 0

def test_powerline_noise(assessor):
    t = np.linspace(0, 5, 2500)
    # Add 60Hz to Lead II (idx 1)
    sig = np.tile(np.sin(2*np.pi*1*t), (12, 1))
    sig[1] += 0.5 * np.sin(2*np.pi*60*t)
    
    report = assessor.assess_quality(sig)
    assert "II 60Hz Noise" in report['issues_detected']
    assert report['lead_scores']['II'] < 100

def test_baseline_wander(assessor):
    t = np.linspace(0, 5, 2500)
    # Add 0.2Hz drift to Lead V1 (idx 6)
    sig = np.tile(np.sin(2*np.pi*1*t), (12, 1))
    sig[6] += 2.0 * np.sin(2*np.pi*0.2*t)
    
    report = assessor.assess_quality(sig)
    assert "V1 Baseline Wander" in report['issues_detected']
    assert "Check skin prep" in str(report['recommendations'])

def test_disconnected_lead(assessor):
    sig = np.zeros((12, 2500))
    # Lead I flat
    sig[0] = np.random.normal(0, 0.001, 2500)
    # Others normal
    t = np.linspace(0, 5, 2500)
    for i in range(1, 12):
        sig[i] = np.sin(2*np.pi*1*t)
        
    report = assessor.assess_quality(sig)
    assert "I Disconnected" in report['issues_detected']
    assert report['lead_scores']['I'] == 0.0
