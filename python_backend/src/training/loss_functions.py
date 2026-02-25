"""
Loss Functions Clínicas para Diagnósticos Multilabel
Conformidade: Balanceamento de classes sem perder sensibilidade
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
import logging


logger = logging.getLogger(__name__)


class WeightedBCEWithLogitsLoss(torch.nn.Module):
    """
    Binary Cross Entropy ponderada para multilabel
    
    Pesos ajustáveis por classe para lidar com desbalanceamento
    """
    
    def __init__(self, class_weights: Optional[Dict[int, float]] = None,
                 pos_weight: float = 2.0):
        super().__init__()
        self.class_weights = class_weights or {}
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) - raw outputs
            targets: (batch_size, num_classes) - binary labels
        
        Returns:
            scalar loss
        """
        # Calcular BCE com pos_weight
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=torch.tensor([self.pos_weight] * logits.shape[-1],
                                   device=logits.device)
        )
        
        # Aplicar class weights
        if self.class_weights:
            class_weight_tensor = torch.tensor(
                [self.class_weights.get(i, 1.0) for i in range(logits.shape[-1])],
                device=logits.device
            )
            bce_loss = bce_loss * class_weight_tensor.mean()
        
        return bce_loss


class FocalLoss(torch.nn.Module):
    """
    Focal Loss para mitigar desbalanceamento extremo
    
    Útil para patologias raras mas críticas em ECG
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss = -α(1-pt)^γ log(pt)
        
        Reduz perda para exemplos fáceis (probabilidade alta/baixa)
        Foca em exemplos difíceis (probabilidade próxima de 0.5)
        """
        probabilities = torch.sigmoid(logits)
        
        # Loss para positivos
        positive_loss = -self.alpha * ((1 - probabilities) ** self.gamma) * \
                       torch.log(probabilities + 1e-7)
        
        # Loss para negativos
        negative_loss = -(1 - self.alpha) * (probabilities ** self.gamma) * \
                       torch.log(1 - probabilities + 1e-7)
        
        # Selecionar based on target
        loss = targets * positive_loss + (1 - targets) * negative_loss
        
        return loss.mean()


class ClinicalCalibrationLoss(torch.nn.Module):
    """
    Penaliza desvios de calibração probabilística
    
    Garante que pred_prob ≈ true_positive_rate
    """
    
    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = n_bins
    
    def forward(self, probabilities: torch.Tensor, 
               targets: torch.Tensor) -> torch.Tensor:
        """
        Expected Calibration Error (ECE)
        """
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=probabilities.device)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        ece = 0.0
        for i in range(self.n_bins):
            bin_mask = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i+1])
            
            if bin_mask.sum() > 0:
                bin_prob = probabilities[bin_mask].mean()
                bin_accuracy = targets[bin_mask].float().mean()
                bin_weight = bin_mask.sum().float() / targets.numel()
                
                ece += bin_weight * torch.abs(bin_prob - bin_accuracy)
        
        return ece
