"""
Advanced ECG Data Augmentation
Melhora generalização e robustez do modelo
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ECGAugmenter:
    """
    Implementa técnicas avançadas de augmentação para ECG
    """
    
    def __init__(self, config):
        self.config = config.augmentation_config
        self.fs = config.signal_config.sampling_rate
        
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Aplica augmentações aleatórias"""
        if not self.config.enabled:
            return signal
            
        aug_signal = signal.copy()
        
        # 1. Baseline Wander (Simula respiração/movimento)
        if np.random.random() < self.config.prob_baseline_wander:
            aug_signal = self._add_baseline_wander(aug_signal)
            
        # 2. Powerline Noise (50/60Hz)
        if np.random.random() < self.config.prob_powerline_noise:
            aug_signal = self._add_powerline_noise(aug_signal)
            
        # 3. EMG Noise (Ruído muscular - alta frequência)
        if np.random.random() < self.config.prob_emg_noise:
            aug_signal = self._add_emg_noise(aug_signal)
            
        # 4. Lead Dropout (Simula eletrodo solto)
        if np.random.random() < self.config.prob_lead_dropout:
            aug_signal = self._lead_dropout(aug_signal)
            
        # 5. Time Warping (Variação de frequência cardíaca artificial)
        if np.random.random() < self.config.prob_time_warp:
            aug_signal = self._time_warp(aug_signal)
            
        # 6. Scaling (Variação de amplitude)
        if np.random.random() < self.config.prob_scaling:
            aug_signal = self._scale_amplitude(aug_signal)
            
        return aug_signal
    
    def _add_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Adiciona oscilação de baixa frequência"""
        n_leads, n_samples = signal.shape
        t = np.linspace(0, n_samples / self.fs, n_samples)
        
        # Gerar baseline aleatória com splines
        num_knots = np.random.randint(3, 10)
        knots_x = np.linspace(0, t[-1], num_knots)
        knots_y = np.random.uniform(
            -self.config.max_baseline_amplitude,
            self.config.max_baseline_amplitude,
            (n_leads, num_knots)
        )
        
        baseline = np.zeros_like(signal)
        for i in range(n_leads):
            cs = CubicSpline(knots_x, knots_y[i])
            baseline[i] = cs(t)
            
        return signal + baseline
    
    def _add_powerline_noise(self, signal: np.ndarray) -> np.ndarray:
        """Adiciona ruído de rede elétrica (50Hz ou 60Hz)"""
        n_leads, n_samples = signal.shape
        t = np.linspace(0, n_samples / self.fs, n_samples)
        freq = 50 if np.random.random() > 0.5 else 60
        
        phase = np.random.uniform(0, 2*np.pi, n_leads)
        amplitude = np.random.uniform(0, self.config.max_noise_amplitude, n_leads)
        
        noise = amplitude[:, None] * np.sin(2 * np.pi * freq * t + phase[:, None])
        return signal + noise
    
    def _add_emg_noise(self, signal: np.ndarray) -> np.ndarray:
        """Adiciona ruído muscular (Gaussiano de alta frequência)"""
        noise = np.random.normal(
            0, 
            self.config.max_noise_amplitude * 0.5, 
            signal.shape
        )
        return signal + noise
    
    def _lead_dropout(self, signal: np.ndarray) -> np.ndarray:
        """Zera leads aleatórios (simula desconexão)"""
        n_leads = signal.shape[0]
        # Nunca dropar todos os leads, manter pelo menos 1
        n_drop = np.random.randint(1, n_leads)
        drop_indices = np.random.choice(n_leads, n_drop, replace=False)
        
        aug_signal = signal.copy()
        aug_signal[drop_indices, :] = 0
        return aug_signal
    
    def _time_warp(self, signal: np.ndarray) -> np.ndarray:
        """Deforma o tempo (esticar/comprimir)"""
        n_leads, n_samples = signal.shape
        
        # Fator de warp
        warp_factor = 1 + np.random.uniform(
            -self.config.max_time_warp_ratio,
            self.config.max_time_warp_ratio
        )
        
        old_x = np.arange(n_samples)
        new_x = np.linspace(0, n_samples - 1, int(n_samples * warp_factor))
        
        aug_signal = np.zeros((n_leads, n_samples))
        
        for i in range(n_leads):
            # Interpolação para novo grid
            resampled = np.interp(new_x, old_x, signal[i])
            
            # Ajustar tamanho de volta para n_samples (crop ou pad)
            if len(resampled) > n_samples:
                start = (len(resampled) - n_samples) // 2
                aug_signal[i] = resampled[start:start+n_samples]
            else:
                pad_left = (n_samples - len(resampled)) // 2
                pad_right = n_samples - len(resampled) - pad_left
                aug_signal[i] = np.pad(resampled, (pad_left, pad_right), 'edge')
                
        return aug_signal

    def _scale_amplitude(self, signal: np.ndarray) -> np.ndarray:
        """Escala amplitude globalmente ou por lead"""
        n_leads = signal.shape[0]
        
        if np.random.random() < 0.5:
            # Global scaling
            scale = 1 + np.random.uniform(
                -self.config.max_scale_ratio,
                self.config.max_scale_ratio
            )
            return signal * scale
        else:
            # Per-lead scaling
            scales = 1 + np.random.uniform(
                -self.config.max_scale_ratio,
                self.config.max_scale_ratio,
                (n_leads, 1)
            )
            return signal * scales

class MixupCutmix:
    """
    Implementação de Mixup e Cutmix para regularização
    """
    def __init__(self, config):
        self.use_mixup = config.training_config.use_mixup
        self.mixup_alpha = config.training_config.mixup_alpha
        self.use_cutmix = config.training_config.use_cutmix
        self.cutmix_alpha = config.training_config.cutmix_alpha
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Aplica Mixup ou Cutmix
        Retorna: x_aug, y_a, y_b, lam
        Loss deve ser calculada como: lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b)
        """
        if not (self.use_mixup or self.use_cutmix):
            return x, y, y, 1.0
            
        # Decidir qual usar (se ambos ativos, 50% chance)
        if self.use_mixup and self.use_cutmix:
            mode = "mixup" if np.random.random() < 0.5 else "cutmix"
        elif self.use_mixup:
            mode = "mixup"
        else:
            mode = "cutmix"
            
        batch_size = x.size(0)
        indices = torch.randperm(batch_size).to(x.device)
        
        y_a = y
        y_b = y[indices]
        
        if mode == "mixup":
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x_aug = lam * x + (1 - lam) * x[indices]
        else:
            # Cutmix para 1D (ECG)
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            seq_len = x.size(2)
            cut_len = int(seq_len * (1 - lam))
            start = np.random.randint(0, seq_len - cut_len)
            
            x_aug = x.clone()
            x_aug[:, :, start:start+cut_len] = x[indices, :, start:start+cut_len]
            
            # Ajustar lambda para proporção real cortada
            lam = 1 - (cut_len / seq_len)
            
        return x_aug, y_a, y_b, lam
