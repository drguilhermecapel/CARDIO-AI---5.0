import pytest
from clinical_criteria.special_alerts import SpecialPatternsDetector

@pytest.fixture
def detector():
    return SpecialPatternsDetector()

def test_hyperkalemia_detection(detector):
    # Tall T waves
    feats = {'max_t_amplitude': 1.5, 'max_t_r_ratio': 1.0, 'qrs_dur': 110}
    res = detector.check_hyperkalemia(feats)
    assert res['match'] is True
    assert "Giant T waves" in res['details']
    assert "URGENT" in res['recommendation']

def test_hypokalemia_detection(detector):
    # Prominent U wave
    feats = {'has_u_wave': True, 'u_wave_amplitude': 0.15, 'max_t_amplitude': 0.4}
    res = detector.check_hypokalemia(feats)
    assert res['match'] is True
    assert "Prominent U waves" in res['details'][0]

def test_pulmonary_embolism_s1q3t3(detector):
    feats = {'hr': 105}
    lead_feats = {
        'I': {'s_amp': -0.2}, # > 1.5mm deep
        'III': {'q_amp': -0.2, 't_amp': -0.1} # Q + Inv T
    }
    res = detector.check_pulmonary_embolism(feats, lead_feats)
    assert res['match'] is True
    assert "S1Q3T3" in res['diagnosis']

def test_takotsubo(detector):
    feats = {
        'min_t_amplitude': -0.6, # Deep inversion
        'qtc': 520, # Long QT
        'has_reciprocal_changes': False
    }
    res = detector.check_takotsubo(feats)
    assert res['match'] is True
    assert "Takotsubo" in res['diagnosis']

def test_brugada_type1(detector):
    feats = {}
    lead_feats = {
        'V1': {'st_elev': 0.25, 't_amp': -0.1}, # Coved STE > 2mm + Inv T
        'V2': {'st_elev': 0.1, 't_amp': 0.1}
    }
    res = detector.check_brugada(feats, lead_feats)
    assert res['match'] is True
    assert "Brugada" in res['diagnosis']
