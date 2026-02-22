import pytest
import numpy as np
from special_patterns import SpecialPatternsDetector

@pytest.fixture
def detector():
    return SpecialPatternsDetector(fs=500)

@pytest.fixture
def mock_signal():
    return np.zeros((12, 5000))

def test_qtc_calculation(detector):
    # HR = 60bpm (RR=1000ms), QT=400ms -> QTc should be 400
    res = detector.calculate_qtc(400, 1000)
    assert res['Bazett'] == 400.0
    assert res['Fridericia'] == 400.0
    
    # HR = 100bpm (RR=600ms), QT=300ms
    # Bazett: 300 / sqrt(0.6) = 387.3
    res = detector.calculate_qtc(300, 600)
    assert 380 < res['Bazett'] < 395

def test_detect_wpw_positive(detector, mock_signal):
    # PR < 120, QRS > 110, Delta Wave
    pr = 100
    qrs = 130
    fiducials = {'Q_start': 1000, 'R_peak': 1050}
    
    # Inject Delta Wave in V4 (idx 9) and V5 (idx 10)
    # Slope initial (20ms) is flat, then steep
    # 20ms = 10 samples at 500Hz
    start = 1000
    mid = 1010
    end = 1050
    
    # V4
    mock_signal[9, start:mid] = np.linspace(0, 0.1, 10) # Slope 0.01
    mock_signal[9, mid:end] = np.linspace(0.1, 1.0, 40) # Slope 0.0225 (2x steeper)
    
    # V5
    mock_signal[10, start:mid] = np.linspace(0, 0.1, 10)
    mock_signal[10, mid:end] = np.linspace(0.1, 1.0, 40)
    
    res = detector.detect_wpw(pr, qrs, mock_signal, fiducials)
    assert res['detected'] is True
    assert 9 in res['delta_wave_leads']

def test_detect_wpw_negative_intervals(detector, mock_signal):
    # Normal PR
    res = detector.detect_wpw(160, 130, mock_signal, {})
    assert res['detected'] is False
    assert "PR interval normal" in res['reason']

def test_detect_brugada_type1(detector, mock_signal):
    # V1 (idx 6) Coved Type
    fiducials = {'J_point': 1000, 'T_peak': 1100}
    
    # J point elevation > 2mm (0.2mV)
    mock_signal[6, 1000] = 0.3
    # Inverted T wave
    mock_signal[6, 1100] = -0.1
    
    res = detector.detect_brugada(mock_signal, fiducials)
    assert res['detected'] is True
    assert res['type'] == "Type 1 (Coved)"

def test_detect_lqts(detector):
    # QTc > 500 -> Severe
    res = detector.detect_lqts(500, 1000, 'Male') # QTc = 500
    assert res['detected'] is True
    assert res['type'] == "Severe LQTS"
    
    # Normal
    res = detector.detect_lqts(400, 1000, 'Male')
    assert res['detected'] is False

def test_detect_sqts(detector):
    # QTc < 340
    res = detector.detect_sqts(300, 1000) # QTc = 300
    assert res['detected'] is True
    assert "High" in res['risk']

def test_detect_early_repol(detector, mock_signal):
    # J point elevation in II, III (Inferior)
    fiducials = {'J_point': 1000}
    
    mock_signal[1, 1000] = 0.15 # II
    mock_signal[2, 1000] = 0.15 # III
    
    res = detector.detect_early_repol(mock_signal, fiducials)
    assert res['detected'] is True
    assert 1 in res['leads']
    assert 2 in res['leads']
