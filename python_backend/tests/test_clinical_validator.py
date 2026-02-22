import pytest
import numpy as np
import pandas as pd
from validation.clinical_validator import ClinicalValidator

@pytest.fixture
def validator():
    return ClinicalValidator()

def test_metrics_calculation(validator):
    # 10 samples, 12 classes
    y_true = np.zeros((10, 12))
    y_true[:, 0] = 1 # All Normal for simplicity
    
    y_pred = y_true.copy()
    y_prob = np.ones((10, 12)) * 0.9
    
    metrics = validator.calculate_metrics(y_pred, y_true, y_prob)
    
    assert "Normal" in metrics
    assert metrics["Normal"]["Sensitivity"] == 1.0
    assert metrics["Normal"]["Specificity"] == 0.0 # No negatives in GT for class 0

def test_subgroup_analysis(validator):
    N = 20
    y_true = np.random.randint(0, 2, (N, 12))
    y_pred = y_true.copy()
    
    metadata = pd.DataFrame({
        'age': [30]*10 + [70]*10,
        'sex': ['Male']*10 + ['Female']*10
    })
    
    subgroups = validator.subgroup_analysis(y_pred, y_true, metadata)
    
    assert "Sex" in subgroups
    assert "Male" in subgroups["Sex"]
    assert subgroups["Sex"]["status"] == "PASS" # Perfect prediction = 0 disparity

def test_stress_test(validator):
    base_signal = np.zeros((12, 5000))
    # Mock model: returns zeros
    model = lambda x: np.zeros((1, 12))
    
    res = validator.stress_test_scenarios(base_signal, model)
    
    assert "Baseline Wander" in res
    assert res["Baseline Wander"]["robust"] is True
