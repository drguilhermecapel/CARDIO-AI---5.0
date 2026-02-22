import pytest
from clinical_criteria.bayesian_assessor import BayesianRiskAssessor

@pytest.fixture
def assessor():
    return BayesianRiskAssessor()

def test_pre_test_probability(assessor):
    # 65yo Male with Typical Angina -> High Risk
    prob = assessor.estimate_pre_test_prob(65, 'Male', ['typical_angina'])
    assert prob > 0.9
    
    # 35yo Female with Non-anginal pain -> Low Risk
    prob = assessor.estimate_pre_test_prob(35, 'Female', ['chest_pain'])
    assert prob < 0.05

def test_post_test_update(assessor):
    # Pre-test 50% (0.5)
    # STEMI (LR+ 100)
    # Odds = 1.0 * 100 = 100
    # Prob = 100/101 ~ 0.99
    post = assessor.calculate_post_test_prob(0.5, "STEMI")
    assert post > 0.98
    
    # Normal ECG (LR- 0.14)
    # Odds = 1.0 * 0.14 = 0.14
    # Prob = 0.14 / 1.14 ~ 0.12
    post = assessor.calculate_post_test_prob(0.5, "Normal")
    assert post < 0.15

def test_timi_score(assessor):
    data = {
        'age': 70, # +1
        'risk_factors': ['htn', 'dm', 'smoke'], # +1
        'history': ['asa_use'], # +1
        'symptoms': ['severe_angina'], # +1
        'labs': {'troponin_positive': True} # +1
    }
    # ECG ST dev (+1)
    # Total 6
    res = assessor.calculate_timi(data, ecg_st_dev=True)
    assert res['score'] == 6
    assert res['risk_level'] == "High"

def test_heart_score(assessor):
    data = {
        'age': 50, # +1
        'risk_factors': ['htn'], # +1
        'history_score': 2, # +2
        'labs': {'troponin_ratio': 0.5} # +0
    }
    # ECG Normal (0)
    res = assessor.calculate_heart(data, ecg_score=0)
    # Total 4 -> Moderate
    assert res['score'] == 4
    assert res['risk_level'] == "Moderate"

def test_full_assessment(assessor):
    patient = {
        'age': 55,
        'sex': 'Male',
        'symptoms': ['atypical_angina'],
        'risk_factors': ['htn', 'dm'],
        'vitals': {'sbp': 130, 'hr': 85},
        'labs': {'troponin_positive': False}
    }
    ecg = {
        'diagnosis': 'Normal',
        'has_st_deviation': False
    }
    
    res = assessor.assess_patient(patient, ecg)
    
    assert 'bayesian_analysis' in res
    assert 'risk_scores' in res
    assert res['bayesian_analysis']['post_test_prob_cad'] < res['bayesian_analysis']['pre_test_prob_cad']
