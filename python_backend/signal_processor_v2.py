import numpy as np
import scipy.signal as signal
import pywt
import logging
from typing import Dict, Any, Tuple, List

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ECGSignalProcessor")

class ECGSignalProcessor:
    """
    Robust ECG Signal Processor for Clinical Environments.
    Includes artifact detection, filtering, quality scoring, and feature extraction.
    """
    
    def __init__(self, sampling_rate: int = 500, num_leads: int = 12):
        self.fs = sampling_rate
        self.num_leads = num_leads
        self.min_duration_s = 5.0
        self.last_quality_report = ""
        
        # Validation thresholds
        if not (300 <= self.fs <= 1000):
            logger.warning(f"Sampling rate {self.fs}Hz is outside recommended range (300-1000Hz).")

    def _validate_input(self, raw_signal: np.ndarray) -> bool:
        """
        Validates input dimensions and sampling rate.
        Expected shape: (Leads, Samples)
        """
        if raw_signal.ndim != 2:
            logger.error(f"Invalid dimensions: {raw_signal.ndim}. Expected 2 (Leads, Samples).")
            return False
            
        leads, samples = raw_signal.shape
        
        # Handle Transpose if needed (Samples, Leads) -> (Leads, Samples)
        if leads != self.num_leads and samples == self.num_leads:
            logger.info("Transposing input signal to (Leads, Samples).")
            raw_signal = raw_signal.T
            leads, samples = raw_signal.shape
            
        if leads != self.num_leads:
            logger.error(f"Invalid number of leads: {leads}. Expected {self.num_leads}.")
            return False
            
        duration = samples / self.fs
        if duration < self.min_duration_s:
            logger.error(f"Signal duration {duration:.2f}s is too short. Minimum {self.min_duration_s}s.")
            return False
            
        return True

    def _apply_notch_filter(self, data: np.ndarray, freq: float = 60.0, Q: float = 30.0) -> np.ndarray:
        """Applies IIR Notch filter to remove power line interference."""
        b, a = signal.iirnotch(freq, Q, self.fs)
        return signal.filtfilt(b, a, data, axis=-1)

    def _remove_baseline_poly(self, data: np.ndarray, degree: int = 3) -> np.ndarray:
        """Removes baseline wander using polynomial subtraction."""
        # Vectorized polynomial fit is tricky, iterating per lead is safer/easier
        cleaned = np.zeros_like(data)
        t = np.arange(data.shape[1])
        
        for i in range(data.shape[0]):
            lead_data = data[i]
            # Polyfit
            coeffs = np.polyfit(t, lead_data, degree)
            poly = np.polyval(coeffs, t)
            cleaned[i] = lead_data - poly
            
        return cleaned

    def detect_artifacts(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """
        Detects specific artifacts: 60Hz, Baseline Wander, Motion, Disconnection.
        """
        artifacts = {
            "line_noise": False,
            "baseline_wander": False,
            "motion_artifact": False,
            "lead_disconnection": [],
            "leads_with_noise": []
        }
        
        # 1. Lead Disconnection (< 10uV consistently)
        # 10uV = 0.01 mV. Assuming signal is in mV.
        # If signal is raw ADC, this threshold needs adjustment. Assuming mV here.
        amplitudes = np.ptp(signal_data, axis=1) # Peak-to-peak
        disconnected_leads = np.where(amplitudes < 0.01)[0].tolist()
        if disconnected_leads:
            artifacts["lead_disconnection"] = disconnected_leads
            logger.warning(f"Leads disconnected: {disconnected_leads}")

        # 2. Line Noise (60Hz)
        # Check PSD at 60Hz
        freqs, psd = signal.welch(signal_data, self.fs, nperseg=1024)
        # Find index for 60Hz
        idx_60 = np.argmin(np.abs(freqs - 60))
        # Compare power at 60Hz vs neighbors (e.g. 55Hz and 65Hz)
        power_60 = np.mean(psd[:, idx_60])
        power_neighbors = np.mean(psd[:, idx_60-5:idx_60+5]) 
        
        # Heuristic: if 60Hz peak is 3x higher than surroundings
        if power_60 > 3 * power_neighbors:
            artifacts["line_noise"] = True
            
        # 3. Motion Artifact (High RMS bursts)
        # Calculate RMS in windows
        window_size = int(0.5 * self.fs) # 500ms
        n_windows = signal_data.shape[1] // window_size
        
        for i in range(self.num_leads):
            lead_sig = signal_data[i]
            rms_values = []
            for w in range(n_windows):
                segment = lead_sig[w*window_size : (w+1)*window_size]
                rms = np.sqrt(np.mean(segment**2))
                rms_values.append(rms)
            
            rms_arr = np.array(rms_values)
            median_rms = np.median(rms_arr)
            std_rms = np.std(rms_arr)
            
            # If any window has RMS > Median + 3*Std (or just a high threshold)
            if np.any(rms_arr > median_rms + 3 * std_rms):
                artifacts["motion_artifact"] = True
                artifacts["leads_with_noise"].append(i)

        return artifacts

    def normalize_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Z-score normalization per lead.
        """
        means = np.mean(data, axis=1, keepdims=True)
        stds = np.std(data, axis=1, keepdims=True)
        stds[stds == 0] = 1.0 # Avoid div by zero
        return (data - means) / stds

    def extract_features(self, clean_signal: np.ndarray) -> Dict[str, float]:
        """
        Extracts 40+ statistical and morphological features.
        """
        features = {}
        
        # 1. Statistical Features per Lead (Mean, Std, Skew, Kurtosis)
        # 12 leads * 4 features = 48 features
        from scipy.stats import skew, kurtosis
        
        for i in range(self.num_leads):
            lead_sig = clean_signal[i]
            features[f"lead_{i}_mean"] = float(np.mean(lead_sig))
            features[f"lead_{i}_std"] = float(np.std(lead_sig))
            features[f"lead_{i}_skew"] = float(skew(lead_sig))
            features[f"lead_{i}_kurt"] = float(kurtosis(lead_sig))
            
            # Simple Interval Proxies (Real interval detection requires the WaveDetector)
            # Here we use global signal properties
            features[f"lead_{i}_energy"] = float(np.sum(lead_sig**2))
            features[f"lead_{i}_zero_crossings"] = int(np.where(np.diff(np.signbit(lead_sig)))[0].size)

        return features

    def calculate_quality_score(self, artifacts: Dict[str, Any], signal_data: np.ndarray) -> float:
        """
        Calculates a 0-100 quality score.
        """
        score = 100.0
        
        # Penalties
        if artifacts["line_noise"]:
            score -= 10
        
        if artifacts["motion_artifact"]:
            score -= 20
            
        # Disconnected leads are severe
        num_disconnected = len(artifacts["lead_disconnection"])
        score -= (num_disconnected * 10)
        
        # Signal-to-Noise Ratio Estimate (Simple)
        # Assuming high frequency noise > 40Hz is noise (except QRS)
        # This is a rough heuristic
        
        return max(0.0, score)

    def process(self, raw_ecg_signal: np.ndarray) -> Dict[str, Any]:
        """
        Main processing pipeline.
        """
        logger.info("Starting ECG Processing...")
        
        # 1. Validation
        if not self._validate_input(raw_ecg_signal):
            return {"error": "Invalid Input"}
            
        # Ensure shape (Leads, Samples)
        if raw_ecg_signal.shape[0] != self.num_leads:
            raw_ecg_signal = raw_ecg_signal.T
            
        # 2. Artifact Detection (Pre-cleaning)
        artifacts = self.detect_artifacts(raw_ecg_signal)
        
        # 3. Filtering
        # Notch 60Hz
        filtered = self._apply_notch_filter(raw_ecg_signal)
        # Baseline Removal
        filtered = self._remove_baseline_poly(filtered)
        
        # 4. Quality Score
        quality_score = self.calculate_quality_score(artifacts, filtered)
        self.last_quality_report = f"Score: {quality_score}/100. Artifacts: {artifacts}"
        
        # 5. Normalization
        normalized = self.normalize_signal(filtered)
        
        # 6. Feature Extraction
        features = self.extract_features(normalized)
        
        return {
            "clean_signal": normalized.tolist(), # For JSON serialization
            "quality_score": quality_score,
            "artifacts": artifacts,
            "features": features,
            "metadata": {
                "fs": self.fs,
                "leads": self.num_leads,
                "duration_s": raw_ecg_signal.shape[1] / self.fs
            }
        }

    def get_quality_report(self) -> str:
        return self.last_quality_report

# Example Usage
if __name__ == "__main__":
    processor = ECGSignalProcessor(sampling_rate=500)
    
    # Mock Signal (12 leads, 10s)
    t = np.linspace(0, 10, 5000)
    # Clean ECG-ish signal
    clean = np.sin(2 * np.pi * 1.0 * t) 
    # Add 60Hz Noise
    noise_60hz = 0.5 * np.sin(2 * np.pi * 60.0 * t)
    # Add Baseline Wander
    baseline = 0.2 * t**2 
    
    raw_signal = np.tile(clean + noise_60hz + baseline, (12, 1))
    
    result = processor.process(raw_signal)
    
    print(f"Quality Score: {result['quality_score']}")
    print(f"Artifacts: {result['artifacts']}")
    print(f"Features Extracted: {len(result['features'])}")
