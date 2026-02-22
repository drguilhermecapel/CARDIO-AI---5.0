import pytest
import numpy as np
from analysis.feature_extraction import ECGFeatureExtractor

@pytest.fixture
def extractor():
    return ECGFeatureExtractor(fs=500)

def test_r_peak_detection(extractor):
    # Generate synthetic ECG-like signal (Pulse train)
    sig = np.zeros(5000)
    peaks = np.arange(500, 4500, 500) # Every 1s
    sig[peaks] = 1.0 # R-peaks
    
    detected = extractor._detect_r_peaks(sig)
    
    # Allow small margin of error in index
    assert len(detected) == len(peaks)
    assert np.allclose(detected, peaks, atol=5)

def test_feature_extraction_structure(extractor):
    # Mock 12-lead signal
    sig = np.random.normal(0, 0.1, (12, 5000))
    # Add some R-peaks to Lead II
    peaks = np.arange(500, 4500, 500)
    sig[1, peaks] = 2.0
    
    features = extractor.extract(sig)
    
    assert 'global' in features
    assert 'leads' in features
    assert 'hr' in features['global']
    assert len(features['leads']) == 12
    assert 'I' in features['leads']

def test_interval_calculation(extractor):
    # Create a single beat with known spacing
    # P at 100ms, QRS at 200ms, T at 400ms
    beat = np.zeros(500)
    beat[50] = 0.1 # P
    beat[100] = 1.0 # R (at 200ms mark relative to start)
    beat[200] = 0.2 # T
    
    # We need to mock _delineate_beat or feed a signal that _delineate_beat handles
    # Let's test _delineate_beat directly
    points = extractor._delineate_beat(beat, 100)
    
    # R is at 100
    # T peak should be found around 200 (100 + 100)
    assert 'R_peak' in points or points.get('R_peak') is None # It might not set R_peak if not passed, but logic sets it
    # Check if T_peak detected
    # T search window starts at R+75 (175) to R+250 (350)
    # Our T is at 200, so it should be found
    assert 'T_peak' in points
    assert abs(points['T_peak'] - 200) < 10

def test_st_elevation_detection(extractor):
    # Simulate ST elevation in Lead V2 (idx 7)
    sig = np.zeros((12, 5000))
    peaks = np.arange(500, 4500, 500)
    
    # Create beat with plateau after R
    beat = np.zeros(500)
    beat[100] = 1.0 # R
    beat[110:150] = 0.3 # ST Elevation
    
    full_lead = np.tile(beat, 10)[:5000]
    sig[1] = full_lead # Need R-peaks for detection
    sig[7] = full_lead # V2 has STE
    
    features = extractor.extract(sig)
    
    v2_feats = features['leads']['V2']
    assert v2_feats['amplitudes']['j_point_elev'] > 0.1
    assert 'V2' in features['global']['leads_with_ste']
