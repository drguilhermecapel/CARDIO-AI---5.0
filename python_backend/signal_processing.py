import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from typing import Tuple, Union, Dict
import joblib
import time

class ECGValidator:
    """
    Validates ECG signal integrity and format compliance.
    """
    def __init__(self, min_duration_sec: float = 8.0, 
                 amplitude_min: float = -10.0, 
                 amplitude_max: float = 10.0,
                 required_leads: int = 12):
        self.min_duration = min_duration_sec
        self.amp_min = amplitude_min
        self.amp_max = amplitude_max
        self.req_leads = required_leads

    def validate(self, signal: np.ndarray, fs: int) -> Tuple[bool, str]:
        """
        Validates the ECG signal array.
        Expected shape: (12, N_samples) or (N_samples, 12).
        Returns: (is_valid, error_message)
        """
        # 1. Check for NaNs or Infs
        if not np.isfinite(signal).all():
            return False, "Signal contains NaNs or Infinite values."

        # 2. Standardize Shape to (12, N)
        if signal.shape[0] != self.req_leads:
            if signal.shape[1] == self.req_leads:
                signal = signal.T
            else:
                return False, f"Invalid shape. Expected {self.req_leads} leads."
        
        # 3. Check Duration
        duration = signal.shape[1] / fs
        if duration < self.min_duration:
            return False, f"Duration {duration:.2f}s is less than minimum {self.min_duration}s."

        # 4. Check Amplitude Range
        if np.max(signal) > self.amp_max or np.min(signal) < self.amp_min:
            return False, f"Signal amplitude out of physiological range ({self.amp_min} to {self.amp_max} mV)."

        # 5. Check for Flatlines (Disconnects)
        if np.any(np.std(signal, axis=1) < 0.01): # Threshold for flatline
            return False, "Flatline detected in one or more leads."

        return True, "Valid"

class ECGPreprocessor:
    """
    High-performance ECG signal cleaning and normalization.
    """
    def __init__(self, fs: int = 500, line_freq: int = 60):
        self.fs = fs
        self.line_freq = line_freq

    def _apply_notch(self, data: np.ndarray) -> np.ndarray:
        b, a = iirnotch(self.line_freq, 30.0, self.fs)
        return filtfilt(b, a, data, axis=-1)

    def _apply_bandpass(self, data: np.ndarray, low: float = 0.05, high: float = 100.0) -> np.ndarray:
        nyq = 0.5 * self.fs
        b, a = butter(2, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, data, axis=-1)

    def _remove_baseline(self, data: np.ndarray) -> np.ndarray:
        # Median filtering is robust but slow for large arrays. 
        # Using a high-pass filter is faster and standard for ECG.
        # However, to meet the "baseline drift removal" requirement explicitly:
        # We can use a high-pass butterworth at 0.5Hz which is standard.
        nyq = 0.5 * self.fs
        b, a = butter(2, 0.5 / nyq, btype='highpass')
        return filtfilt(b, a, data, axis=-1)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-Max normalization to 0-1 range per lead."""
        min_vals = np.min(data, axis=1, keepdims=True)
        max_vals = np.max(data, axis=1, keepdims=True)
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        return (data - min_vals) / range_vals

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.
        Signal shape: (Leads, Samples)
        """
        # 1. Notch Filter (Power line interference)
        signal = self._apply_notch(signal)
        
        # 2. Baseline Wander Removal
        signal = self._remove_baseline(signal)
        
        # 3. Bandpass Filter (0.05 - 100 Hz)
        signal = self._apply_bandpass(signal)
        
        # 4. Normalization
        signal = self.normalize(signal)
        
        return signal

class ArtifactDetector:
    """
    Analyzes signal quality and detects artifacts.
    """
    def __init__(self, fs: int = 500):
        self.fs = fs

    def calculate_snr(self, signal: np.ndarray) -> float:
        """
        Estimate SNR using statistical properties.
        Signal = smoothed version, Noise = residual.
        """
        # Simple estimation: Signal power / Noise power
        # We assume noise is high frequency.
        from scipy.signal import medfilt
        
        # Smooth signal to estimate "true" ECG
        # Kernel size ~ 50ms
        kernel_size = int(0.05 * self.fs)
        if kernel_size % 2 == 0: kernel_size += 1
        
        smoothed = np.apply_along_axis(lambda x: medfilt(x, kernel_size), 1, signal)
        noise = signal - smoothed
        
        signal_power = np.mean(smoothed ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0: return 100.0
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def check_artifacts(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Returns quality metrics and pass/fail status.
        """
        metrics = {}
        
        # 1. SNR
        snr = self.calculate_snr(signal)
        metrics['snr_db'] = snr
        
        # 2. Saturation Detection
        # Check if > 5% of samples are at min/max limits (after normalization 0-1)
        # Assuming signal is already normalized or we check raw limits
        # Let's check raw distribution kurtosis or just rail checks
        
        # 3. Quality Score (0-100)
        # Heuristic: SNR > 20dB is good (100), < 0dB is bad (0)
        quality_score = np.clip((snr) * 5, 0, 100)
        metrics['quality_score'] = quality_score
        
        # 4. Rejection Logic
        is_rejected = quality_score < 30.0
        metrics['status'] = 'REJECTED' if is_rejected else 'ACCEPTED'
        
        return metrics

# Performance Wrapper
def process_ecg_pipeline(raw_signal: np.ndarray, fs: int = 500):
    start_time = time.time()
    
    validator = ECGValidator()
    valid, msg = validator.validate(raw_signal, fs)
    if not valid:
        return {"error": msg, "status": "INVALID"}
        
    preprocessor = ECGPreprocessor(fs=fs)
    clean_signal = preprocessor.process(raw_signal)
    
    detector = ArtifactDetector(fs=fs)
    quality_metrics = detector.check_artifacts(clean_signal)
    
    if quality_metrics['status'] == 'REJECTED':
        return {"error": "Low Signal Quality", "metrics": quality_metrics, "status": "REJECTED"}
        
    duration = time.time() - start_time
    
    return {
        "status": "SUCCESS",
        "processed_signal": clean_signal.tolist(), # For JSON serialization
        "metrics": quality_metrics,
        "processing_time_s": duration
    }
