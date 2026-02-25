"""
CARDIO-AI 5.0 - Configuration Module
Conformidade: ISO 13485, ISO 14971, IEC 62366
Versão: 1.0.0
Data: 2026-02-24
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum
import json
from pathlib import Path


class DeviceMode(Enum):
    """Modos operacionais do dispositivo"""
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    CLINICAL = "clinical"


class DiagnosticClass(Enum):
    """Classes diagnósticas suportadas"""
    # Ritmo normal
    NORMAL_SINUS_RHYTHM = "NSR"
    
    # Arritmias
    ATRIAL_FIBRILLATION = "AF"
    ATRIAL_FLUTTER = "AFL"
    SUPRAVENTRICULAR_TACHYCARDIA = "SVT"
    VENTRICULAR_TACHYCARDIA = "VT"
    VENTRICULAR_FIBRILLATION = "VF"
    PREMATURE_ATRIAL_CONTRACTION = "PAC"
    PREMATURE_VENTRICULAR_CONTRACTION = "PVC"
    
    # Bloqueios de condução
    FIRST_DEGREE_AV_BLOCK = "AVB1"
    SECOND_DEGREE_AV_BLOCK_TYPE1 = "AVB2_1"
    SECOND_DEGREE_AV_BLOCK_TYPE2 = "AVB2_2"
    THIRD_DEGREE_AV_BLOCK = "AVB3"
    RIGHT_BUNDLE_BRANCH_BLOCK = "RBBB"
    LEFT_BUNDLE_BRANCH_BLOCK = "LBBB"
    LEFT_ANTERIOR_FASCICULAR_BLOCK = "LAFB"
    LEFT_POSTERIOR_FASCICULAR_BLOCK = "LPFB"
    
    # Alterações ST/T
    ST_ELEVATION = "STE"
    ST_DEPRESSION = "STD"
    T_WAVE_INVERSION = "TWI"
    
    # Prolongamento de intervalo
    PROLONGED_QT = "LQTS"
    PROLONGED_PR = "LONG_PR"
    
    # Hipertrofia
    LEFT_VENTRICULAR_HYPERTROPHY = "LVH"
    RIGHT_VENTRICULAR_HYPERTROPHY = "RVH"
    
    # Infarto
    ACUTE_MYOCARDIAL_INFARCTION = "AMI"
    INFERIOR_MYOCARDIAL_INFARCTION = "IMI"
    ANTERIOR_MYOCARDIAL_INFARCTION = "AMI_ANT"
    LATERAL_MYOCARDIAL_INFARCTION = "LMI"
    
    # Outras
    WOLFF_PARKINSON_WHITE = "WPW"
    BRUGADA_PATTERN = "BRUGADA"
    EARLY_REPOLARIZATION = "ER"
    HYPERTROPHIC_CARDIOMYOPATHY = "HCM"
    LEFT_VENTRICULAR_DYSFUNCTION = "LVD"


@dataclass
class ECGSignalConfig:
    """Configuração de aquisição de sinal ECG"""
    sampling_rate: int = 500  # Hz
    duration_seconds: int = 10  # segundos
    num_leads: int = 12
    leads: List[str] = field(default_factory=lambda: [
        'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ])
    mains_frequency: int = 50  # Hz (Brasil)
    amplitude_range: Tuple[float, float] = (-30000, 30000)  # microvolts
    
    @property
    def samples_per_ecg(self) -> int:
        return self.sampling_rate * self.duration_seconds
    
    @property
    def total_samples(self) -> int:
        return self.num_leads * self.samples_per_ecg


@dataclass
class PreprocessingConfig:
    """Configuração de pré-processamento de sinal"""
    # Filtragem
    highpass_freq: float = 0.5  # Hz
    lowpass_freq: float = 150.0  # Hz
    notch_freq: float = 50.0  # Hz
    notch_quality: float = 30.0
    
    # Normalização
    normalize_method: str = "z-score"  # "z-score", "min-max"
    normalize_per_lead: bool = True
    
    # Detecção de ruído
    max_noise_threshold: float = 3.0  # desvios padrão
    min_signal_quality: float = 0.85  # 85%
    
    # Baseline wandering removal
    remove_baseline: bool = True
    baseline_polynomial_order: int = 5
    
    # Interpolação de dados ruins
    interpolate_missing: bool = True
    max_missing_samples: int = 50  # amostras


@dataclass
class WaveDetectionConfig:
    """Configuração para detecção de ondas"""
    # Limites clínicos esperados (segundos)
    min_pr_interval: float = 0.12
    max_pr_interval: float = 0.20
    min_qrs_duration: float = 0.08
    max_qrs_duration: float = 0.12
    min_qt_interval: float = 0.30
    max_qt_interval: float = 0.70
    
    # Parâmetros de detecção
    p_wave_prominence_threshold: float = 0.1
    qrs_prominence_threshold: float = 0.3
    t_wave_prominence_threshold: float = 0.1
    
    # Tolerância de detecção (ms)
    p_wave_tolerance_ms: int = 50
    qrs_tolerance_ms: int = 30
    t_wave_tolerance_ms: int = 50
    
    # Validação
    min_detection_confidence: float = 0.80


@dataclass
class ModelConfig:
    """Configuração da arquitetura de modelo"""
    # Arquitetura
    model_type: str = "ssm_hybrid"  # "cnn", "transformer", "ssm", "ssm_hybrid"
    
    # SSM (State Space Model) - Mamba
    ssm_state_size: int = 256
    ssm_expand_factor: int = 2
    ssm_num_layers: int = 6
    
    # CNN (se usado)
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [5, 5, 3, 3])
    
    # Transformer (se usado)
    transformer_hidden_dim: int = 512
    transformer_num_heads: int = 8
    transformer_num_layers: int = 6
    
    # Dropout
    dropout_rate: float = 0.3
    
    # Saída
    num_diagnoses: int = len(DiagnosticClass)
    multilabel_output: bool = True
    
    # Calibração
    use_temperature_scaling: bool = True
    calibration_set_size: float = 0.1


@dataclass
class AugmentationConfig:
    """Configuração de Data Augmentation Avançado"""
    enabled: bool = True
    # Probabilidades
    prob_baseline_wander: float = 0.5
    prob_powerline_noise: float = 0.3
    prob_emg_noise: float = 0.3
    prob_lead_dropout: float = 0.1
    prob_time_warp: float = 0.2
    prob_scaling: float = 0.5
    
    # Parâmetros
    max_baseline_amplitude: float = 0.5
    max_noise_amplitude: float = 0.1
    max_time_warp_ratio: float = 0.2  # ±20% speed change
    max_scale_ratio: float = 0.2      # ±20% amplitude change


@dataclass
class TrainingConfig:
    """Configuração de treinamento"""
    # Dados
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    
    # Otimização
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    warmup_epochs: int = 5
    
    # Loss function
    loss_type: str = "weighted_bce"  # "bce", "weighted_bce", "focal", "asymmetric"
    class_weights: Dict[str, float] = field(default_factory=lambda: {
        "VENTRICULAR_FIBRILLATION": 5.0,  # Alta prioridade (risco de vida)
        "VENTRICULAR_TACHYCARDIA": 4.0,   # Alta prioridade
        "PREMATURE_VENTRICULAR_CONTRACTION": 2.0,
        "ACUTE_MYOCARDIAL_INFARCTION": 4.0,
        "ATRIAL_FIBRILLATION": 2.0
    })
    
    # Advanced Training Techniques
    use_ema: bool = True  # Exponential Moving Average dos pesos
    ema_decay: float = 0.9999
    
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    
    label_smoothing: float = 0.05
    
    # Validação
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_auroc"
    early_stopping_min_delta: float = 1e-4
    
    # Device
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 2


@dataclass
class ValidationConfig:
    """Configuração de validação clínica"""
    # Métricas mínimas obrigatórias
    min_auroc: float = 0.999
    min_sensitivity: float = 0.95
    min_specificity: float = 0.95
    min_ppv: float = 0.92
    min_npv: float = 0.98
    
    # Análise de subgrupos
    validate_by_age_groups: bool = True
    age_groups: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 30), (30, 50), (50, 70), (70, 100)
    ])
    
    validate_by_sex: bool = True
    validate_by_comorbidities: bool = True
    
    # Validação externa
    external_validation_required: bool = True
    external_validation_min_samples: int = 1000
    
    # Testes específicos
    test_noise_robustness: bool = True
    test_artifact_robustness: bool = True
    test_long_duration_ecg: bool = True


@dataclass
class RegulatoryConfig:
    """Configuração de conformidade regulatória"""
    # Aplicável para Brasil
    regulatory_standard: str = "CE_MDR"  # "CE_MDR", "FDA_510k"
    iso_13485_compliant: bool = True
    iso_14971_compliant: bool = True
    iec_62366_compliant: bool = True
    
    # Rastreabilidade
    enable_audit_trail: bool = True
    enable_result_signing: bool = True
    
    # Data retention
    retain_results_days: int = 2555  # 7 anos
    retain_logs_days: int = 1825  # 5 anos
    
    # Risk classification
    device_classification: str = "Class_II"
    predicate_devices: List[str] = field(default_factory=list)


@dataclass
class CardioAIConfig:
    """Configuração global do CARDIO-AI 5.0"""
    # Modo de operação
    mode: DeviceMode = DeviceMode.CLINICAL
    version: str = "5.0.0"
    device_id: str = "CARDIO-AI-5.0-BR-001"
    
    # Sub-configurações
    signal_config: ECGSignalConfig = field(default_factory=ECGSignalConfig)
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    wave_detection_config: WaveDetectionConfig = field(default_factory=WaveDetectionConfig)
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    regulatory_config: RegulatoryConfig = field(default_factory=RegulatoryConfig)
    
    # Paths
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    logs_dir: Path = Path("./logs")
    results_dir: Path = Path("./results")
    
    def __post_init__(self):
        """Criar diretórios necessários"""
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Converter configuração para dicionário"""
        return {
            'mode': self.mode.value,
            'version': self.version,
            'device_id': self.device_id,
            'signal_config': self.signal_config.__dict__,
            'preprocessing_config': self.preprocessing_config.__dict__,
            'wave_detection_config': self.wave_detection_config.__dict__,
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__,
            'validation_config': self.validation_config.__dict__,
            'regulatory_config': self.regulatory_config.__dict__,
        }
    
    def save(self, path: Path):
        """Salvar configuração em JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4, default=str)
    
    @classmethod
    def load(cls, path: Path) -> 'CardioAIConfig':
        """Carregar configuração de JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Instância global de configuração
default_config = CardioAIConfig(mode=DeviceMode.CLINICAL)
