import pytest
import numpy as np
from models.advanced_ensemble import AdvancedStackingEnsemble

@pytest.fixture
def ensemble():
    return AdvancedStackingEnsemble(num_classes=3, input_shape=(100, 12))

def test_model_building(ensemble):
    cnn = ensemble.build_cnn()
    assert cnn.output_shape == (None, 3)
    
    lstm = ensemble.build_lstm_attention()
    assert lstm.output_shape == (None, 3)
    
    transformer = ensemble.build_transformer()
    assert transformer.output_shape == (None, 3)

def test_training_flow(ensemble):
    # Small dataset
    N = 20
    X = np.random.randn(N, 100, 12).astype(np.float32)
    y = np.eye(3)[np.random.choice(3, N)]
    
    ensemble.fit(X, y, epochs=1, batch_size=4)
    
    assert ensemble.is_fitted is True
    assert len(ensemble.base_models) == 3
    
    # Prediction
    preds = ensemble.predict(X[:2])
    assert preds.shape == (2, 3)
    # Check probabilities sum to 1
    assert np.allclose(np.sum(preds, axis=1), 1.0)
