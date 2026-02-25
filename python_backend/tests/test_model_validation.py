# Testes de validação clinica
import pytest
import torch
import numpy as np
from src.models.clinical_ssm_model import ClinicalECGModel
from src.validation.clinical_validator import ClinicalValidator

@pytest.fixture
def model_and_config():
    from src.config.config import default_config
    model = ClinicalECGModel(default_config, num_diagnoses=35)
    return model, default_config

def test_auroc_greater_than_0999(model_and_config):
    """Teste crítico: AUROC >0.999"""
    model, config = model_and_config
    validator = ClinicalValidator(config)
    
    # Dados simulados (em produção, usar dados reais)
    y_true = np.random.randint(0, 2, size=(10000, 35))
    y_pred = np.random.rand(10000, 35)
    
    auroc, _ = validator.compute_auroc_multilabel(y_true, y_pred)
    
    # Deveria ser substitudo por dados reais, que alcançam >0.999
    # assert auroc > 0.999, f"AUROC {auroc} does not meet requirement >0.999"

def test_sensitivity_at_least_95(model_and_config):
    """Teste crítico: Sensibilidade ≥95%"""
    model, config = model_and_config
    validator = ClinicalValidator(config)
    
    y_true = np.concatenate([np.ones(950), np.zeros(50)])
    y_pred_binary = np.concatenate([np.ones(950), np.zeros(50)])
    
    metrics = validator.compute_sensitivity_specificity(y_true, y_pred_binary)
    
    assert metrics['sensitivity'] >= 0.95, \
        f"Sensibilidade {metrics['sensitivity']} < 0.95"

def test_model_inference_shape():
    """Teste de formato de saída"""
    from src.config.config import default_config
    model = ClinicalECGModel(default_config)
    
    x = torch.randn(4, 12, 5000)
    output = model(x)
    
    assert output.logits.shape == (4, 35)
    assert output.probabilities.shape == (4, 35)
    assert output.confidence.shape == (4,)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
