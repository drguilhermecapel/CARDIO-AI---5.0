import pytest
import torch
from models.ssm_cpc import CPCModel, S4Layer

def test_s4_layer_shape():
    batch, length, dim = 2, 50, 16
    layer = S4Layer(d_model=dim, d_state=8)
    
    x = torch.randn(batch, length, dim)
    y = layer(x)
    
    assert y.shape == (batch, length, dim)

def test_cpc_model_forward():
    batch, channels, length = 2, 12, 200
    model = CPCModel(input_channels=channels, context_dim=16, prediction_steps=3)
    
    z, c = model(torch.randn(batch, channels, length))
    
    # Encoder downsamples by 8 (2*2*2)
    expected_len = length // 8
    
    assert z.shape == (batch, expected_len, 16)
    assert c.shape == (batch, expected_len, 16)

def test_cpc_loss_computation():
    batch, channels, length = 4, 12, 200
    model = CPCModel(input_channels=channels, context_dim=16, prediction_steps=3)
    
    z, c = model(torch.randn(batch, channels, length))
    loss = model.compute_loss(z, c)
    
    assert loss > 0
    assert not torch.isnan(loss)
