import pytest
import numpy as np
from explainability.ecg_explainer import ECGExplainer
from ensemble_analyzer import EnsembleECGAnalyzer

@pytest.fixture
def explainer():
    ensemble = EnsembleECGAnalyzer() # Dummy models
    return ECGExplainer(ensemble_analyzer=ensemble)

@pytest.fixture
def mock_ecg():
    return np.random.randn(5000, 12).astype(np.float32)

def test_lead_attribution_sum(explainer, mock_ecg):
    # Test if lead attributions sum to ~1.0
    explanation = explainer.explain_diagnosis(mock_ecg)
    contribs = explanation['lead_contribution']
    
    total = sum(contribs.values())
    assert 0.99 < total < 1.01
    assert len(contribs) == 12

def test_temporal_peak(explainer, mock_ecg):
    explanation = explainer.explain_diagnosis(mock_ecg)
    peak = explanation['temporal_peak']
    
    assert 'start_ms' in peak
    assert 'end_ms' in peak
    assert peak['start_ms'] < peak['end_ms']

def test_visualization_generation(explainer, mock_ecg):
    explanation = explainer.explain_diagnosis(mock_ecg)
    img_bytes = explainer.visualize_explanation(mock_ecg, explanation)
    
    assert len(img_bytes) > 0
    assert img_bytes.startswith(b'\x89PNG') # PNG signature

def test_integrated_gradients_shape(explainer, mock_ecg):
    model = list(explainer.ensemble.models.values())[0]
    ig = explainer._compute_integrated_gradients(model, mock_ecg, class_idx=0, steps=5)
    
    assert ig.shape == mock_ecg.shape
