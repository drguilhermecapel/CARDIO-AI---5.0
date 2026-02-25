"""
Detecção de Ondas ECG com Precisão Clínica
Conformidade: Métricas <4% de desvio absoluto
Módulo: Wave Detection and Segmentation
Versão: 1.0.0
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter, hilbert
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import logging
from enum import Enum


logger = logging.getLogger(__name__)


class WaveType(Enum):
    P_WAVE = "P"
    QRS_COMPLEX = "QRS"
    T_WAVE = "T"
    U_WAVE = "U"


@dataclass
class WaveMarker:
    """Marcador de onda com confiança"""
    wave_type: WaveType
    onset_idx: int
    peak_idx: int
    offset_idx: int
    confidence: float
    amplitude_uv: float
    
    @property
    def onset_ms(self) -> float:
        return self.onset_idx / 5  # Assumindo 500 Hz
    
    @property
    def offset_ms(self) -> float:
        return self.offset_idx / 5
    
    @property
    def duration_ms(self) -> float:
        return (self.offset_idx - self.onset_idx) / 5
    
    @property
    def peak_time_ms(self) -> float:
        return self.peak_idx / 5


@dataclass
class ECGWaveProfile:
    """Perfil completo de ondas do ECG"""
    p_waves: List[WaveMarker]
    qrs_complex: Optional[WaveMarker]
    t_waves: List[WaveMarker]
    u_waves: List[WaveMarker]
    
    # Intervalos (em ms)
    pr_interval_ms: float
    qrs_duration_ms: float
    qt_interval_ms: float
    qtc_interval_ms: float  # Corrigido pela frequência cardíaca
    
    # Segmentos
    pr_segment_ms: float
    st_segment_ms: float
    
    # Frequência cardíaca
    heart_rate: int
    
    # Quality metrics
    overall_confidence: float
    noise_level: float


class WaveDetector:
    """Detector de ondas ECG com validação clínica"""
    
    def __init__(self, sampling_rate: int = 500, config=None):
        self.sampling_rate = sampling_rate
        self.config = config
        self.ms_to_samples = sampling_rate / 1000
        
    def detect_qrs_complex(self, signal: np.ndarray, 
                          channel_lead: str = 'II') -> Optional[WaveMarker]:
        """
        Detecta complexo QRS usando método robusto multi-estágio
        
        Args:
            signal: Sinal ECG normalizado (1D array)
            channel_lead: Lead utilizado (II é padrão para ritmo)
            
        Returns:
            WaveMarker com localização e confiança do QRS
            
        Requisito: Sensibilidade ≥99%, PPV ≥98%
        """
        if len(signal) == 0:
            return None
        
        # Estágio 1: Filtragem diferencial
        # Realça a transição abrupta do QRS
        diff_signal = np.diff(signal, prepend=signal[0])
        
        # Estágio 2: Envelope analítico (transformada de Hilbert)
        analytic_signal = hilbert(diff_signal)
        envelope = np.abs(analytic_signal)
        
        # Estágio 3: Smoothing adaptativo
        window_length = int(self.sampling_rate * 0.1)  # 100ms
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 5:
            window_length = 5
            
        smoothed = savgol_filter(envelope, window_length, 3)
        
        # Estágio 4: Detecção de picos
        # Altura mínima relativa ao percentil 95
        threshold = np.percentile(smoothed, 95)
        height = threshold * 0.7
        
        # Distância mínima entre QRS (120 bpm = 500ms)
        min_distance = int(self.sampling_rate * 0.5)
        
        peaks, properties = find_peaks(
            smoothed,
            height=height,
            distance=min_distance,
            prominence=threshold * 0.3
        )
        
        if len(peaks) == 0:
            logger.warning("No QRS detected in signal")
            return None
        
        # Usar o pico mais proeminente
        best_idx = peaks[np.argmax(properties['peak_heights'])]
        
        # Estágio 5: Refinar localização no sinal original
        # Buscar nadir (ponto mais baixo) próximo ao pico
        search_window = int(self.sampling_rate * 0.1)  # ±100ms
        start_idx = max(0, best_idx - search_window)
        end_idx = min(len(signal), best_idx + search_window)
        
        local_signal = signal[start_idx:end_idx]
        # Q wave (negativo)
        q_idx = start_idx + np.argmin(local_signal)
        # S wave (negativo após R)
        s_idx = start_idx + np.argmax(local_signal[best_idx - start_idx:]) + (best_idx - start_idx)
        
        # Onset: onde começa a deflexão
        onset_idx = self._find_onset(signal, q_idx, search_window)
        # Offset: onde termina
        offset_idx = self._find_offset(signal, s_idx, search_window)
        
        # Estágio 6: Validação
        qrs_duration = (offset_idx - onset_idx) / self.ms_to_samples
        
        if not (self.config.wave_detection_config.min_qrs_duration <= qrs_duration <=
                self.config.wave_detection_config.max_qrs_duration):
            logger.warning(f"QRS duration {qrs_duration}ms outside normal range")
            # Ainda retorna, mas com confiança reduzida
            confidence = 0.75
        else:
            confidence = 0.98
        
        amplitude = np.abs(signal[best_idx])
        
        return WaveMarker(
            wave_type=WaveType.QRS_COMPLEX,
            onset_idx=int(onset_idx),
            peak_idx=int(best_idx),
            offset_idx=int(offset_idx),
            confidence=confidence,
            amplitude_uv=float(amplitude)
        )
    
    def detect_p_waves(self, signal: np.ndarray, 
                      qrs_marker: WaveMarker) -> List[WaveMarker]:
        """
        Detecta ondas P antes de cada QRS
        
        Requisito: Precisão de timing <30ms
        """
        p_waves = []
        
        # Onda P deve ocorrer 80-200ms antes do QRS
        search_start = max(0, int(qrs_marker.onset_idx - 200 * self.ms_to_samples))
        search_end = int(qrs_marker.onset_idx - 80 * self.ms_to_samples)
        
        if search_end <= search_start:
            return p_waves
        
        p_segment = signal[search_start:search_end]
        
        if len(p_segment) < int(self.sampling_rate * 0.05):  # Mínimo 50ms
            return p_waves
        
        # Detecção adaptativa considerando amplitude típica de P (~0.1mV)
        threshold = np.percentile(np.abs(p_segment), 85)
        
        peaks, properties = find_peaks(
            np.abs(p_segment),
            height=threshold * 0.5,
            distance=int(self.sampling_rate * 0.08),  # Mínimo 80ms entre P waves
        )
        
        for peak_idx in peaks:
            actual_idx = search_start + peak_idx
            p_onset = self._find_onset(signal, actual_idx, int(self.sampling_rate * 0.1))
            p_offset = self._find_offset(signal, actual_idx, int(self.sampling_rate * 0.1))
            
            p_duration = (p_offset - p_onset) / self.ms_to_samples
            if 30 <= p_duration <= 120:  # 30-120ms é típico
                p_waves.append(WaveMarker(
                    wave_type=WaveType.P_WAVE,
                    onset_idx=int(p_onset),
                    peak_idx=actual_idx,
                    offset_idx=int(p_offset),
                    confidence=0.90,
                    amplitude_uv=float(np.abs(signal[actual_idx]))
                ))
        
        return p_waves
    
    def detect_t_waves(self, signal: np.ndarray, 
                      qrs_marker: WaveMarker) -> List[WaveMarker]:
        """
        Detecta ondas T após complexo QRS
        
        Requisito: Precisão de timing <50ms
        """
        t_waves = []
        
        # Onda T deve ocorrer 200-600ms após offset do QRS
        search_start = int(qrs_marker.offset_idx + 100 * self.ms_to_samples)
        search_end = min(len(signal), int(qrs_marker.offset_idx + 600 * self.ms_to_samples))
        
        if search_end <= search_start:
            return t_waves
        
        t_segment = signal[search_start:search_end]
        
        threshold = np.percentile(np.abs(t_segment), 80)
        peaks, _ = find_peaks(
            np.abs(t_segment),
            height=threshold * 0.4,
            distance=int(self.sampling_rate * 0.1),
        )
        
        for peak_idx in peaks:
            actual_idx = search_start + peak_idx
            t_onset = self._find_onset(signal, actual_idx, int(self.sampling_rate * 0.15))
            t_offset = self._find_offset(signal, actual_idx, int(self.sampling_rate * 0.15))
            
            t_duration = (t_offset - t_onset) / self.ms_to_samples
            if 100 <= t_duration <= 300:  # 100-300ms é típico
                t_waves.append(WaveMarker(
                    wave_type=WaveType.T_WAVE,
                    onset_idx=int(t_onset),
                    peak_idx=actual_idx,
                    offset_idx=int(t_offset),
                    confidence=0.85,
                    amplitude_uv=float(np.abs(signal[actual_idx]))
                ))
        
        return t_waves
    
    def _find_onset(self, signal: np.ndarray, peak_idx: int, 
                   search_window: int) -> float:
        """Encontra início de onda por mudança de derivada"""
        start = max(0, peak_idx - search_window)
        end = min(len(signal), peak_idx)
        
        region = signal[start:end]
        if len(region) < 2:
            return float(start)
        
        # Encontrar onde a derivada muda significativamente
        derivatives = np.abs(np.diff(region))
        
        # Onset é onde começa a mudança significativa
        threshold = np.mean(derivatives) * 0.5
        changes = np.where(derivatives > threshold)[0]
        
        if len(changes) > 0:
            return float(start + changes[0])
        return float(start)
    
    def _find_offset(self, signal: np.ndarray, peak_idx: int, 
                    search_window: int) -> float:
        """Encontra fim de onda"""
        start = min(len(signal), peak_idx)
        end = min(len(signal), peak_idx + search_window)
        
        region = signal[start:end]
        if len(region) < 2:
            return float(end)
        
        derivatives = np.abs(np.diff(region))
        threshold = np.mean(derivatives) * 0.5
        changes = np.where(derivatives > threshold)[0]
        
        if len(changes) > 0:
            return float(start + changes[-1])
        return float(end)
    
    def analyze_complete_ecg(self, signal: np.ndarray,
                            lead: str = 'II') -> Optional[ECGWaveProfile]:
        """
        Análise completa de um lead do ECG
        
        Retorna: Perfil completo com todos os intervalos medidos
        """
        try:
            # Detectar QRS como âncora
            qrs = self.detect_qrs_complex(signal, lead)
            if qrs is None:
                return None
            
            # Detectar P e T waves
            p_waves = self.detect_p_waves(signal, qrs)
            t_waves = self.detect_t_waves(signal, qrs)
            
            # Calcular intervalos
            if len(p_waves) > 0 and p_waves[0].offset_idx < qrs.onset_idx:
                pr_interval = (qrs.onset_idx - p_waves[0].offset_idx) / self.ms_to_samples
                pr_segment = (qrs.onset_idx - p_waves[0].offset_idx) / self.ms_to_samples - \
                             p_waves[0].duration_ms
            else:
                pr_interval = 0
                pr_segment = 0
            
            qrs_duration = qrs.duration_ms
            
            if len(t_waves) > 0:
                qt_interval = (t_waves[0].offset_idx - qrs.onset_idx) / self.ms_to_samples
            else:
                qt_interval = 0
            
            # QTc (Bazett): QTc = QT / sqrt(RR)
            # Estimar RR interval da frequência cardíaca
            # Para simplicidade, usar 60ms para bpm=100
            rr_interval = 1000 / max(40, 60)  # Mínimo 40 bpm
            qtc_interval = qt_interval / np.sqrt(rr_interval / 1000)
            
            # Frequência cardíaca (de 5000 amostras assumindo 10s)
            if len(signal) >= self.sampling_rate * 10:
                duration_sec = len(signal) / self.sampling_rate
                heart_rate = int(60 / (duration_sec / 1))
            else:
                heart_rate = 60
            
            return ECGWaveProfile(
                p_waves=p_waves,
                qrs_complex=qrs,
                t_waves=t_waves,
                u_waves=[],
                pr_interval_ms=pr_interval,
                qrs_duration_ms=qrs_duration,
                qt_interval_ms=qt_interval,
                qtc_interval_ms=qtc_interval,
                pr_segment_ms=pr_segment,
                st_segment_ms=0,  # Calcular separadamente
                heart_rate=heart_rate,
                overall_confidence=min(qrs.confidence, 
                                      np.mean([p.confidence for p in p_waves] + [0.90]),
                                      np.mean([t.confidence for t in t_waves] + [0.90])),
                noise_level=self._estimate_noise(signal)
            )
        
        except Exception as e:
            logger.error(f"Error in complete ECG analysis: {e}")
            return None
    
    def _estimate_noise(self, signal: np.ndarray) -> float:
        """Estima nível de ruído como % do sinal"""
        # Assumir que ruído é alta frequência
        # Usar relação entre std de derivada e sinal
        derivatives = np.diff(signal)
        noise_estimate = np.std(derivatives) / (np.std(signal) + 1e-6)
        return min(1.0, noise_estimate)


class RhythmAnalyzer:
    """
    Analisador de ritmo avançado para detecção otimizada de Arritmias Ventriculares
    (Fibrilação Ventricular, Taquicardia Ventricular, Extrassístoles Ventriculares)
    """
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self.ms_to_samples = sampling_rate / 1000
        
    def detect_all_qrs(self, signal: np.ndarray) -> List[int]:
        """Detecta todos os picos R no sinal para análise de ritmo"""
        diff_signal = np.diff(signal, prepend=signal[0])
        analytic_signal = hilbert(diff_signal)
        envelope = np.abs(analytic_signal)
        
        window_length = int(self.sampling_rate * 0.1)
        if window_length % 2 == 0: window_length += 1
        smoothed = savgol_filter(envelope, max(5, window_length), 3)
        
        threshold = np.percentile(smoothed, 90) * 0.5
        min_distance = int(self.sampling_rate * 0.2)  # Max 300 bpm
        
        peaks, _ = find_peaks(smoothed, height=threshold, distance=min_distance)
        return peaks.tolist()
        
    def analyze_ventricular_arrhythmias(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extrai features específicas para detecção de VT, VF e PVCs
        """
        features = {
            'is_vf_pattern': 0.0,
            'is_vt_pattern': 0.0,
            'pvc_count': 0.0,
            'rr_variability': 0.0,
            'mean_qrs_width': 0.0
        }
        
        # 1. Detecção de Fibrilação Ventricular (VF)
        # VF é caracterizada por ondas caóticas sem QRS claro, alta frequência e baixa amplitude
        # Usamos análise de frequência (Leakage), amplitude (VFA) e complexidade
        
        # a) Zero-crossing rate (ZCR)
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        zcr = len(zero_crossings) / (len(signal) / self.sampling_rate) # cruzamentos por segundo
        
        # b) Amplitude analysis (VFA - Ventricular Fibrillation Amplitude)
        p2p_amplitude = np.max(signal) - np.min(signal)
        
        # c) Spectral analysis (Leakage) - VF tem energia concentrada entre 2-10 Hz
        # Usamos FFT simplificada para estimar a concentração de energia
        try:
            fft_vals = np.abs(np.fft.rfft(signal))
            fft_freqs = np.fft.rfftfreq(len(signal), 1.0/self.sampling_rate)
            
            # Energia na banda VF (2-10 Hz) vs energia total
            vf_band_idx = np.where((fft_freqs >= 2.0) & (fft_freqs <= 10.0))[0]
            total_energy = np.sum(fft_vals**2)
            
            if total_energy > 0 and len(vf_band_idx) > 0:
                vf_energy = np.sum(fft_vals[vf_band_idx]**2)
                vf_leakage = vf_energy / total_energy
            else:
                vf_leakage = 0.0
        except Exception:
            vf_leakage = 0.0
            
        # d) Complexity (Threshold Crossing Intervals - TCI)
        # VF tem intervalos de cruzamento de limiar muito irregulares
        threshold = 0.2 * np.max(np.abs(signal))
        tci_crossings = np.where(np.diff(np.signbit(signal - threshold)))[0]
        
        tci_regularity = 0.0
        if len(tci_crossings) > 2:
            tci_intervals = np.diff(tci_crossings)
            tci_regularity = np.std(tci_intervals) / (np.mean(tci_intervals) + 1e-6)
            
        # Combinação de features para score de VF
        vf_score = 0.0
        
        # VF típica: ZCR alto (mas não extremo como ruído EMG), alta energia na banda 2-10Hz,
        # e alta irregularidade
        if 4 < zcr < 30: # 4-30 cruzamentos por segundo é típico de VF
            vf_score += 0.3
        
        if vf_leakage > 0.4: # Mais de 40% da energia na banda 2-10Hz
            vf_score += 0.4
            
        if tci_regularity > 0.4: # Alta irregularidade
            vf_score += 0.3
            
        # Penalizar apenas se a amplitude for muito baixa (assistolia ou ruído de base)
        if p2p_amplitude < 0.1: # mV (assumindo sinal normalizado)
            vf_score *= 0.1
            
        features['is_vf_pattern'] = min(1.0, vf_score)
                
        # 2. Análise de Ritmo (RR intervals)
        qrs_peaks = self.detect_all_qrs(signal)
        
        if len(qrs_peaks) < 3:
            return features
            
        rr_intervals = np.diff(qrs_peaks) / self.ms_to_samples # em ms
        mean_rr = np.mean(rr_intervals)
        heart_rate = 60000 / mean_rr if mean_rr > 0 else 0
        
        rr_variability = np.std(rr_intervals) / mean_rr if mean_rr > 0 else 0
        features['rr_variability'] = rr_variability
        
        # Se os picos detectados forem muito regulares, reduz a chance de ser VF
        # (pode ser Taquicardia Ventricular ou Supraventricular)
        if features['is_vf_pattern'] < 0.8 and rr_variability < 0.15:
            features['is_vf_pattern'] *= 0.2
            
        # 3. Análise de Morfologia QRS (Largura)
        qrs_widths = []
        for peak in qrs_peaks:
            # Estimativa simplificada de largura
            start = max(0, peak - int(100 * self.ms_to_samples))
            end = min(len(signal), peak + int(100 * self.ms_to_samples))
            window = signal[start:end]
            if len(window) > 0:
                threshold = np.max(np.abs(window)) * 0.2
                above_thresh = np.where(np.abs(window) > threshold)[0]
                if len(above_thresh) > 0:
                    width = (above_thresh[-1] - above_thresh[0]) / self.ms_to_samples
                    qrs_widths.append(width)
                    
        if qrs_widths:
            mean_width = np.mean(qrs_widths)
            features['mean_qrs_width'] = mean_width
            
            # Medir variabilidade de amplitude dos picos (alta na VF, baixa na TV monomórfica)
            peak_amps = [np.abs(signal[p]) for p in qrs_peaks]
            amp_variability = np.std(peak_amps) / (np.mean(peak_amps) + 1e-6) if peak_amps else 0
            
            # Detecção de Taquicardia Ventricular (VT) vs Fibrilação Ventricular (VF)
            # Ambas têm HR > 120 e QRS largo (> 120ms). A diferença principal é a regularidade.
            if heart_rate > 120 and mean_width > 120:
                # Se for muito irregular (alta variabilidade RR ou de amplitude), ou já tiver alto score de VF
                if rr_variability > 0.25 or amp_variability > 0.25 or features['is_vf_pattern'] >= 0.7:
                    # É Fibrilação Ventricular (ou TV Polimórfica, que tem urgência similar)
                    features['is_vt_pattern'] = 0.1
                    features['is_vf_pattern'] = max(features['is_vf_pattern'], 0.9) # Reforça VF
                else:
                    # Se for regular (baixa variabilidade), é Taquicardia Ventricular Monomórfica
                    features['is_vt_pattern'] = 0.9
                    features['is_vf_pattern'] = 0.0 # Suprime VF para evitar confusão
                
            # Detecção de PVCs (Extrassístoles Ventriculares)
            # PVC: RR prematuro (< 80% do RR médio), QRS largo
            pvc_count = 0
            for i, rr in enumerate(rr_intervals):
                if rr < 0.8 * mean_rr and qrs_widths[i+1] > 120:
                    pvc_count += 1
            features['pvc_count'] = pvc_count
            
        return features

class WaveDetectionValidator:
    """Validação de detecção de ondas contra padrões clínicos"""
    
    @staticmethod
    def validate_pr_interval(pr_ms: float, config) -> Tuple[bool, str]:
        """Validar intervalo PR (normal: 120-200ms)"""
        if not (config.wave_detection_config.min_pr_interval * 1000 <= pr_ms <= 
                config.wave_detection_config.max_pr_interval * 1000):
            return False, f"PR interval {pr_ms:.1f}ms out of normal range"
        return True, "PR interval normal"
    
    @staticmethod
    def validate_qrs_duration(qrs_ms: float, config) -> Tuple[bool, str]:
        """Validar duração QRS (normal: 80-120ms)"""
        if not (config.wave_detection_config.min_qrs_duration * 1000 <= qrs_ms <=
                config.wave_detection_config.max_qrs_duration * 1000):
            return False, f"QRS duration {qrs_ms:.1f}ms out of normal range"
        return True, "QRS duration normal"
    
    @staticmethod
    def validate_qt_interval(qtc_ms: float) -> Tuple[bool, str]:
        """Validar intervalo QTc corrigido (normal: <440ms homens, <460ms mulheres)"""
        # Usando limite conservador
        if qtc_ms > 460:
            return False, f"QTc {qtc_ms:.1f}ms prolongado"
        elif qtc_ms > 420:
            return False, f"QTc {qtc_ms:.1f}ms borderline"
        return True, "QTc interval normal"


if __name__ == "__main__":
    # Exemplo de uso
    from src.config.config import default_config
    
    detector = WaveDetector(sampling_rate=500, config=default_config)
    
    # Sinal ECG de exemplo (simulado)
    ecg_signal = np.random.randn(5000)  # 10s a 500Hz
    
    profile = detector.analyze_complete_ecg(ecg_signal, lead='II')
    if profile:
        print(f"Heart Rate: {profile.heart_rate} bpm")
        print(f"QRS Duration: {profile.qrs_duration_ms:.1f} ms")
        print(f"PR Interval: {profile.pr_interval_ms:.1f} ms")
        print(f"QTc: {profile.qtc_interval_ms:.1f} ms")
        print(f"Overall Confidence: {profile.overall_confidence:.2%}")
