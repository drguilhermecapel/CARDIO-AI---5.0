import pytest
from clinical_criteria.mi_staging import MIStagingClassifier

@pytest.fixture
def classifier():
    return MIStagingClassifier()

def test_hyperacute_stage(classifier):
    feats = {
        'max_st_elevation': 0.2,
        'max_t_amplitude': 1.5, # Very tall T
        'has_pathological_q': False
    }
    res = classifier.classify_stage(feats)
    assert res['stage'] == "Hyperacute"
    assert res['confidence'] >= 0.9

def test_acute_stage(classifier):
    feats = {
        'max_st_elevation': 0.2,
        'max_t_amplitude': 0.4,
        'min_t_amplitude': -0.2, # Inverting
        'has_pathological_q': True # Q wave started
    }
    res = classifier.classify_stage(feats)
    assert res['stage'] == "Acute"

def test_subacute_stage(classifier):
    feats = {
        'max_st_elevation': 0.05, # Resolved
        'min_t_amplitude': -0.5, # Deep inversion
        'has_pathological_q': True
    }
    res = classifier.classify_stage(feats)
    assert res['stage'] == "Subacute"

def test_old_stage(classifier):
    feats = {
        'max_st_elevation': 0.0,
        'min_t_amplitude': 0.1, # Normalized T
        'has_pathological_q': True
    }
    res = classifier.classify_stage(feats)
    assert res['stage'] == "Old/Scar"

def test_clinical_correlation(classifier):
    feats = {
        'max_st_elevation': 0.3,
        'max_t_amplitude': 1.2,
        'has_pathological_q': False
    }
    # Clinical info matches Hyperacute
    res = classifier.classify_stage(feats, {'onset_hours': 1})
    assert res['stage'] == "Hyperacute"
    assert "Matches clinical timeline" in res['reasoning']
