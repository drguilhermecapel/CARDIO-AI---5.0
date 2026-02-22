import pytest
from clinical_criteria.genetic_arrhythmia import GeneticArrhythmiaDetector

@pytest.fixture
def detector():
    return GeneticArrhythmiaDetector()

def test_wpw_detection(detector):
    feats = {'pr_interval': 90, 'qrs_dur': 120, 'has_delta_wave': True}
    res = detector.check_wpw(feats)
    assert res['match'] is True
    assert "Delta wave" in res['details']

def test_brugada_type1(detector):
    lead_feats = {
        'V1': {'st_elev': 0.25, 'st_morphology': 'coved'},
        'V2': {'st_elev': 0.1}
    }
    res = detector.check_brugada_detailed(lead_feats)
    assert res['match'] is True
    assert "Type 1" in res['diagnosis']

def test_long_qt_syndrome(detector):
    # Male with QTc 490
    feats = {'qtc': 490, 'sex': 'Male'}
    res = detector.check_long_qt_syndrome(feats)
    assert res['match'] is True
    assert res['probability'] == "High"

def test_short_qt_syndrome(detector):
    feats = {'qtc': 320}
    res = detector.check_short_qt_syndrome(feats)
    assert res['match'] is True
    assert "Short QT Syndrome" in res['diagnosis']

def test_early_repolarization(detector):
    # Inferior leads J-point elev + notch
    lead_feats = {
        'II': {'j_point_elev': 0.15, 'has_qrs_notch': True},
        'III': {'j_point_elev': 0.12, 'has_qrs_notch': True},
        'aVF': {'j_point_elev': 0.1, 'has_qrs_notch': True}
    }
    res = detector.check_early_repolarization(lead_feats)
    assert res['match'] is True
    assert "Inferior" in res['details'][0]
