import pytest
from clinical_criteria.decision_support import ClinicalDecisionSupport

@pytest.fixture
def cds():
    return ClinicalDecisionSupport()

def test_stemi_activation(cds):
    # Contiguous Inferior leads
    ecg = {'leads_with_ste': ['II', 'III', 'aVF']}
    res = cds.evaluate(ecg, {})
    
    assert res['protocol'] == "STEMI Protocol"
    assert "ACTIVATE CATH LAB" in res['action']
    assert res['status'] == "CRITICAL"

def test_nstemi_protocol(cds):
    # T-inversion + Positive Troponin
    ecg = {'leads_with_t_inv': ['V1', 'V2', 'V3'], 'leads_with_ste': []}
    pat = {'troponin_status': 'Positive'}
    
    res = cds.evaluate(ecg, pat)
    
    assert res['protocol'] == "NSTEMI Protocol"
    assert "Dual Antiplatelet" in res['action']

def test_pericarditis_mimic(cds):
    # Diffuse STE + PR depression
    ecg = {
        'leads_with_ste': ['I', 'II', 'V2', 'V3', 'V4', 'V5', 'V6'], 
        'has_pr_depression': True
    }
    res = cds.evaluate(ecg, {})
    
    assert "Pericarditis" in res['protocol']
    assert "Urgent Echo" in res['target_time']

def test_unstable_angina(cds):
    # Ischemia signs + Symptoms (Troponin Unknown/Neg)
    ecg = {'leads_with_std': ['V4', 'V5', 'V6']}
    pat = {'symptoms': ['chest_pain'], 'troponin_status': 'Negative'}
    
    res = cds.evaluate(ecg, pat)
    
    assert res['protocol'] == "Unstable Angina Protocol"
    assert res['status'] == "HIGH_RISK"
