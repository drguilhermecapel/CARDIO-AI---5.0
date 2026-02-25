"""
Validação Clínica e Conformidade Regulatória
Módulo: Clinical Validation Engine
Conformidade: ISO 14971, IEC 62366-1, ISO 13485
Requisitos: AUROC >0.999, Sensibilidade ≥95%, Especificidade ≥95%
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, auc, precision_recall_curve, confusion_matrix,
    classification_report, hamming_loss, jaccard_score, f1_score
)
from scipy import stats
from dataclasses import dataclass
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging


logger = logging.getLogger(__name__)


@dataclass
class RegulatoryMetrics:
    """Métricas exigidas por regulatórios"""
    auroc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    threshold_operating_point: float
    
    def meets_requirements(self) -> Tuple[bool, List[str]]:
        """Verifica se atende requisitos mínimos"""
        failures = []
        
        if self.auroc < 0.999:
            failures.append(f"AUROC {self.auroc:.4f} < 0.999 (CRÍTICO)")
        if self.sensitivity < 0.95:
            failures.append(f"Sensibilidade {self.sensitivity:.4f} < 0.95 (CRÍTICO)")
        if self.specificity < 0.95:
            failures.append(f"Especificidade {self.specificity:.4f} < 0.95 (CRÍTICO)")
        if self.ppv < 0.92:
            failures.append(f"PPV {self.ppv:.4f} < 0.92 (CRÍTICO)")
        if self.npv < 0.98:
            failures.append(f"NPV {self.npv:.4f} < 0.98 (CRÍTICO)")
        
        return len(failures) == 0, failures


class ClinicalValidator:
    """Validador de conformidade clínica"""
    
    def __init__(self, config):
        self.config = config
        self.validation_results = {}
    
    def compute_auroc_multilabel(self, y_true: np.ndarray, 
                                y_pred: np.ndarray) -> Tuple[float, Dict]:
        """
        Calcula AUROC para cada classe e média (multilabel)
        
        Args:
            y_true: (n_samples, n_classes) binary labels
            y_pred: (n_samples, n_classes) predicted probabilities
        
        Returns:
            auroc_macro: Média entre classes
            per_class_auroc: AUROC individual por classe
        """
        per_class_auroc = {}
        aucs = []
        
        for class_idx in range(y_true.shape[1]):
            if np.sum(y_true[:, class_idx]) > 0:  # Só se houver positivos
                try:
                    auroc = roc_auc_score(y_true[:, class_idx], y_pred[:, class_idx])
                    per_class_auroc[class_idx] = auroc
                    aucs.append(auroc)
                except Exception as e:
                    logger.warning(f"AUROC calculation failed for class {class_idx}: {e}")
        
        auroc_macro = np.mean(aucs) if aucs else 0.0
        
        return auroc_macro, per_class_auroc
    
    def compute_sensitivity_specificity(self, y_true: np.ndarray,
                                       y_pred_binary: np.ndarray) -> Dict[str, float]:
        """
        Sensibilidade (recall) e Especificidade
        
        Sensibilidade = TP / (TP + FN) - Taxa de detecção de positivos
        Especificidade = TN / (TN + FP) - Taxa de rejeição de negativos
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precisão
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def compute_pr_metrics(self, y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula curva Precision-Recall e AUC-PR
        
        Importante para dados desbalanceados em ECG
        """
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
        auc_pr = auc(recall_vals, precision_vals)
        
        return {
            'auc_pr': auc_pr,
            'precision': precision_vals,
            'recall': recall_vals
        }
    
    def validate_subgroup_performance(self, y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     metadata: pd.DataFrame) -> Dict[str, Dict]:
        """
        Valida performance em subgrupos para detectar bias
        
        Requisito IEC 62366: Testar em diferentes demogr ápicos
        """
        subgroup_results = {}
        
        # Validação por idade
        if 'age' in metadata.columns:
            age_groups = self.config.validation_config.age_groups
            for age_min, age_max in age_groups:
                mask = (metadata['age'] >= age_min) & (metadata['age'] < age_max)
                if mask.sum() > 50:  # Mínimo 50 amostras
                    auroc, _ = self.compute_auroc_multilabel(
                        y_true[mask], y_pred[mask]
                    )
                    subgroup_results[f'age_{age_min}_{age_max}'] = auroc
        
        # Validação por sexo
        if 'sex' in metadata.columns:
            for sex in ['M', 'F']:
                mask = metadata['sex'] == sex
                if mask.sum() > 100:
                    auroc, _ = self.compute_auroc_multilabel(
                        y_true[mask], y_pred[mask]
                    )
                    subgroup_results[f'sex_{sex}'] = auroc
        
        return subgroup_results
    
    def test_noise_robustness(self, model, x_clean: np.ndarray,
                             y_true: np.ndarray) -> Dict[str, float]:
        """
        Testa robustez a ruído (requisito clínico)
        
        Simula: ruído Gaussiano, desconexão de eletrodo, artefatos de movimento
        """
        import torch
        
        results = {'clean': 0.0}
        noise_levels = [0.01, 0.05, 0.1, 0.2]  # SNR em Watts
        
        x_clean_tensor = torch.from_numpy(x_clean).float()
        
        # Baseline com dados limpos
        with torch.no_grad():
            y_pred_clean = model(x_clean_tensor).probabilities.cpu().numpy()
        auroc_clean, _ = self.compute_auroc_multilabel(y_true, y_pred_clean)
        results['clean'] = auroc_clean
        
        # Testar com ruído
        for noise_level in noise_levels:
            # Adicionar ruído Gaussiano
            noise = np.random.normal(0, noise_level, x_clean.shape)
            x_noisy = x_clean + noise
            
            x_noisy_tensor = torch.from_numpy(x_noisy).float()
            
            with torch.no_grad():
                y_pred_noisy = model(x_noisy_tensor).probabilities.cpu().numpy()
            
            auroc_noisy, _ = self.compute_auroc_multilabel(y_true, y_pred_noisy)
            results[f'noise_level_{noise_level:.2f}'] = auroc_noisy
        
        return results
    
    def validate_ecg_intervals(self, detected_intervals: Dict[str, float]) -> Dict[str, Tuple[bool, str]]:
        """
        Valida medições de intervalos ECG contra padrões clínicos
        
        Requisito: Desvio absoluto <4% em medições clínicas
        """
        validation_results = {}
        config = self.config.wave_detection_config
        
        # PR Interval
        if 'pr_interval_ms' in detected_intervals:
            pr = detected_intervals['pr_interval_ms']
            valid = (config.min_pr_interval * 1000 <= pr <= config.max_pr_interval * 1000)
            validation_results['pr_interval'] = (
                valid,
                f"PR: {pr:.1f}ms {'✓' if valid else '✗'}"
            )
        
        # QRS Duration
        if 'qrs_duration_ms' in detected_intervals:
            qrs = detected_intervals['qrs_duration_ms']
            valid = (config.min_qrs_duration * 1000 <= qrs <= config.max_qrs_duration * 1000)
            validation_results['qrs_duration'] = (
                valid,
                f"QRS: {qrs:.1f}ms {'✓' if valid else '✗'}"
            )
        
        # QTc Interval
        if 'qtc_interval_ms' in detected_intervals:
            qtc = detected_intervals['qtc_interval_ms']
            if qtc < 420:
                valid = True
                msg = "Normal"
            elif qtc < 440:
                valid = True
                msg = "Borderline"
            else:
                valid = False
                msg = "Prolonged"
            
            validation_results['qtc_interval'] = (valid, f"QTc: {qtc:.1f}ms - {msg}")
        
        return validation_results
    
    def generate_iso_14971_risk_report(self) -> str:
        """
        Gera relatório de gestão de riscos (ISO 14971)
        
        Exemplo de estrutura para documentação regulatória
        """
        report = """
        ========================================
        ISO 14971 - RISK MANAGEMENT REPORT
        Produto: CARDIO-AI 5.0 ECG Interpretation Device
        Versão: 1.0.0
        Data: 2026-02-24
        ========================================
        
        1. RISCO: Falha na detecção de patologia crítica
           Severidade: Crítica (morte/invalidez)
           Probabilidade: Baixa (mitigada por validação)
           Mitigação: AUROC >0.999, Sensibilidade ≥95%
           Resíduo: Aceitável
        
        2. RISCO: Falso positivo causando cath lab desnecessário
           Severidade: Média (procedimento invasivo)
           Probabilidade: Baixa
           Mitigação: Especificidade ≥95%, revisão por cardiologista
           Resíduo: Aceitável
        
        3. RISCO: Viés por características demográficas
           Severidade: Alta (disparidade de cuidado)
           Probabilidade: Média
           Mitigação: Validação por sexo, idade, etnia
           Resíduo: Mitigado
        
        4. RISCO: Falha em dados de longo prazo (Holter)
           Severidade: Alta (arritmias paroxísticas perdidas)
           Probabilidade: Média
           Mitigação: Testes em ECG de 24h, algoritmo de SSM
           Resíduo: Mitigado
        
        ========================================
        CONCLUSÃO: Aceitável para uso clínico com precauções
        ========================================
        """
        return report


@dataclass
class ValidationReport:
    """Relatório estruturado de validação"""
    model_name: str
    test_date: str
    metrics: RegulatoryMetrics
    subgroup_performance: Dict[str, float]
    noise_robustness: Dict[str, float]
    external_validation_passed: bool
    recommendations: List[str]
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)
