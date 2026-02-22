import pytest
import numpy as np
from quality.lead_placement import LeadPlacementDetector

@pytest.fixture
def detector():
    return LeadPlacementDetector()

def test_missing_leads(detector):
    sig = np.random.randn(12, 1000)
    sig[0] = 0.0 # Flatline Lead I
    
    res = detector.validate_placement(sig)
    assert "Missing Leads: I" in res['alerts'][0]

def test_ra_la_reversal(detector):
    sig = np.zeros((12, 1000))
    # Lead I Negative
    sig[0] = -1.0 * np.abs(np.random.randn(1000)) 
    # aVR Positive
    sig[3] = 1.0 * np.abs(np.random.randn(1000))
    
    res = detector.check_ra_la_reversal(sig)
    assert res['detected'] is True
    assert "RA/LA Reversal" in res['description']

def test_einthoven_law(detector):
    sig = np.zeros((12, 1000))
    sig[0] = np.random.randn(1000) # I
    sig[2] = np.random.randn(1000) # III
    sig[1] = sig[0] + sig[2] # II = I + III (Correct)
    
    res = detector.check_einthoven_law(sig)
    assert res['detected'] is False
    
    # Violation
    sig[1] = sig[0] + sig[2] + 5.0 # Add huge error
    res = detector.check_einthoven_law(sig)
    assert res['detected'] is True

def test_precordial_progression(detector):
    sig = np.zeros((12, 1000))
    # V1 (idx 6) > V2 (idx 7) < V3 (idx 8)
    sig[6] = 2.0 * np.sin(np.linspace(0, 10, 1000))
    sig[7] = 0.5 * np.sin(np.linspace(0, 10, 1000))
    sig[8] = 1.5 * np.sin(np.linspace(0, 10, 1000))
    
    res = detector.check_precordial_progression(sig)
    assert res['detected'] is True
    assert "V1/V2 Reversal" in res['details'][0]
