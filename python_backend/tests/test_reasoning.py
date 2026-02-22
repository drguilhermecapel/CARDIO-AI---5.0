import pytest
import numpy as np
from reasoning.diagnostic_engine import DiagnosticReasoningEngine

@pytest.fixture
def engine():
    return DiagnosticReasoningEngine()

def test_justification_generation(engine):
    diag = "STEMI"
    feats = {'max_st_elevation': 0.2}
    criteria = {'Sgarbossa': {'match': True}}
    lead_contrib = {'V2': 0.4, 'V3': 0.3, 'V4': 0.1}
    
    text = engine._generate_justification(diag, feats, criteria, lead_contrib)
    
    assert "Significant ST elevation detected in leads V2, V3" in text
    assert "Sgarbossa Criteria: POSITIVE" in text

def test_differentials_sorting(engine):
    probs = [0.1, 0.8, 0.1]
    names = ["Normal", "STEMI", "AFib"]
    
    diffs = engine._get_differentials(probs, names)
    
    assert diffs[0]['diagnosis'] == "STEMI"
    assert diffs[0]['confidence_level'] == "High"
    assert diffs[1]['diagnosis'] in ["Normal", "AFib"]

def test_full_analysis_flow(engine):
    ecg = np.random.randn(5000, 12).astype(np.float32)
    meta = {'age': 50}
    
    res = engine.analyze(ecg, meta)
    
    assert 'primary_diagnosis' in res
    assert 'clinical_reasoning' in res
    assert 'recommendation' in res
    assert len(res['differentials']) == 3
