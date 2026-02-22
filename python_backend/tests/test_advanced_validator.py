import pytest
import numpy as np
from clinical_criteria.advanced_validator import AdvancedCriteriaValidator

@pytest.fixture
def validator():
    return AdvancedCriteriaValidator()

def test_sgarbossa_criteria(validator):
    # Mock LBBB signal with Concordant STE in V5 (idx 10)
    sig = np.zeros((12, 5000))
    fiducials = {'J_point': 1000, 'Q_start': 950, 'R_peak': 980, 'S_peak': 1020}
    
    # V5: Positive QRS (R wave), STE 0.2mV
    sig[10, 1000] = 0.2 
    
    # Should detect if LBBB is True
    res = validator.check_sgarbossa(sig, fiducials, is_lbbb=True)
    assert res['match'] is True
    assert "Concordant STE" in res['details'][0]
    
    # Should fail if not LBBB
    res = validator.check_sgarbossa(sig, fiducials, is_lbbb=False)
    assert res['match'] is False

def test_lvh_criteria(validator):
    sig = np.zeros((12, 5000))
    # Sokolow-Lyon: S_V1 (2.0) + R_V5 (2.0) = 4.0 > 3.5
    fiducials = {
        'amplitudes': {'V1_S': 2.0, 'V5_R': 2.0, 'V6_R': 1.5, 'aVL_R': 0.5}
    }
    
    res = validator.check_lvh_criteria(sig, fiducials)
    assert res['Sokolow_Lyon']['match'] is True
    assert res['Sokolow_Lyon']['value'] == 4.0

def test_heart_score(validator):
    history = {'score': 2, 'atherosclerosis': True}
    res = validator.calculate_heart_score(history, ecg_score=2, age=70, risk_factors=3, troponin=0.05)
    
    # H=2, E=2, A=2 (70yo), R=2 (3 RF), T=0
    # Total = 8 -> High Risk
    assert res['score'] == 8
    assert res['risk_category'] == "High"

def test_de_winter(validator):
    # V3 (idx 8): STD -0.2mV, Tall T 0.6mV
    sig = np.zeros((12, 5000))
    fiducials = {'J_point': 1000, 'T_peak': 1100}
    
    sig[8, 1000] = -0.2
    sig[8, 1100] = 0.6
    
    res = validator.check_de_winter(sig, fiducials)
    assert res['match'] is True
    assert "LAD Occlusion" in res['risk']
