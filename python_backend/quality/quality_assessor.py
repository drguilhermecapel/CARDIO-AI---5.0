import numpy as np
import scipy.signal as signal
from typing import Dict, List, Any, Tuple

class QualityAssessor:
    """
    Advanced ECG Quality Assessment Module.
    Detects specific noise types and provides actionable recommendations.
    """
    
    def __init__(self, fs: int = 500):
        self.fs = fs
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def _calculate_psd(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return signal.welch(sig, self.fs, nperseg=1024)

    def detect_powerline_noise(self, sig: np.ndarray, freqs: np.ndarray, psd: np.ndarray) -> bool:
        """
        Detects 50Hz or 60Hz interference.
        """
        # Check 60Hz
        idx_60 = np.argmin(np.abs(freqs - 60))
        power_60 = np.mean(psd[idx_60-2:idx_60+3])
        
        # Check 50Hz
        idx_50 = np.argmin(np.abs(freqs - 50))
        power_50 = np.mean(psd[idx_50-2:idx_50+3])
        
        # Baseline (neighboring frequencies)
        baseline_60 = np.mean(psd[idx_60-10:idx_60-5])
        baseline_50 = np.mean(psd[idx_50-10:idx_50-5])
        
        # Threshold: Peak > 3x Baseline
        if power_60 > 5 * baseline_60: return True, "60Hz"
        if power_50 > 5 * baseline_50: return True, "50Hz"
        
        return False, None

    def detect_emg_noise(self, sig: np.ndarray, freqs: np.ndarray, psd: np.ndarray) -> bool:
        """
        Detects Muscle Tremor / EMG (Broadband high frequency > 30Hz).
        """
        # EMG typically 20-500Hz. ECG main power < 40Hz.
        # Check power in 70-150Hz band (avoiding 60Hz harmonics if possible)
        
        mask_high = (freqs > 70) & (freqs < 150)
        power_high = np.mean(psd[mask_high])
        
        mask_ecg = (freqs > 1) & (freqs < 40)
        power_ecg = np.mean(psd[mask_ecg])
        
        # Ratio of High Freq Noise to ECG Signal
        ratio = power_high / (power_ecg + 1e-9)
        
        # Threshold determined empirically
        return ratio > 0.05

    def detect_baseline_wander(self, sig: np.ndarray) -> bool:
        """
        Detects significant baseline drift (< 0.5Hz).
        """
        # Low pass filter at 0.5Hz
        sos = signal.butter(2, 0.5, 'low', fs=self.fs, output='sos')
        baseline = signal.sosfilt(sos, sig)
        
        # Check amplitude of baseline
        # If baseline moves more than 1mV (assuming signal is mV)
        # or if std dev is high
        
        drift_range = np.ptp(baseline)
        return drift_range > 1.5 # > 1.5mV drift is significant

    def check_lead_connection(self, sig: np.ndarray) -> bool:
        """
        Detects flatline or disconnected lead.
        """
        # Check if signal is essentially zero or just noise
        amp = np.ptp(sig)
        if amp < 0.05: # < 50uV
            return False # Disconnected
        return True

    def assess_quality(self, leads_signal: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes 12-lead ECG for quality issues.
        
        Args:
            leads_signal: (12, Samples) array
            
        Returns:
            Quality Report Dictionary
        """
        issues = []
        lead_scores = {}
        recommendations = []
        
        total_leads = leads_signal.shape[0]
        
        for i in range(total_leads):
            sig = leads_signal[i]
            lead_name = self.leads[i] if i < 12 else f"Lead_{i+1}"
            
            # 1. Connection
            if not self.check_lead_connection(sig):
                issues.append(f"{lead_name} Disconnected")
                recommendations.append(f"Check electrode {lead_name}.")
                lead_scores[lead_name] = 0.0
                continue
                
            # PSD Calculation
            freqs, psd = self._calculate_psd(sig)
            
            # 2. Powerline
            has_pl, pl_type = self.detect_powerline_noise(sig, freqs, psd)
            if has_pl:
                issues.append(f"{lead_name} {pl_type} Noise")
                if "Check grounding" not in recommendations:
                    recommendations.append("Check device grounding and nearby electrical equipment.")
            
            # 3. EMG
            if self.detect_emg_noise(sig, freqs, psd):
                issues.append(f"{lead_name} EMG/Tremor")
                if "Ask patient to relax" not in recommendations:
                    recommendations.append("Ask patient to relax limbs and stop talking.")
            
            # 4. Baseline
            if self.detect_baseline_wander(sig):
                issues.append(f"{lead_name} Baseline Wander")
                if "Check skin prep" not in recommendations:
                    recommendations.append("Check skin preparation and electrode contact.")
            
            # Score Calculation (Simple)
            score = 100
            if has_pl: score -= 10
            if self.detect_emg_noise(sig, freqs, psd): score -= 20
            if self.detect_baseline_wander(sig): score -= 15
            lead_scores[lead_name] = max(0, score)

        # Aggregate
        avg_score = np.mean(list(lead_scores.values())) if lead_scores else 0
        
        reliability = "High"
        if avg_score < 50: reliability = "Unusable"
        elif avg_score < 80: reliability = "Low"
        elif avg_score < 90: reliability = "Medium"
        
        return {
            "overall_score": round(avg_score, 1),
            "reliability": reliability,
            "issues_detected": list(set(issues)),
            "recommendations": list(set(recommendations)),
            "lead_scores": lead_scores,
            "is_acceptable": avg_score >= 70
        }

# Example
if __name__ == "__main__":
    assessor = QualityAssessor()
    
    # Mock Signal
    t = np.linspace(0, 5, 2500)
    sig = np.zeros((12, 2500))
    
    # Lead I: Clean
    sig[0] = np.sin(2*np.pi*1*t)
    
    # Lead II: 60Hz Noise
    sig[1] = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*60*t)
    
    # Lead III: EMG Noise (High Freq)
    sig[2] = np.sin(2*np.pi*1*t) + 0.2*np.random.normal(0, 1, 2500)
    
    # Lead V1: Baseline Wander
    sig[6] = np.sin(2*np.pi*1*t) + 2.0*np.sin(2*np.pi*0.2*t)
    
    # Lead V6: Disconnected
    sig[11] = np.random.normal(0, 0.01, 2500)
    
    report = assessor.assess_quality(sig)
    import json
    print(json.dumps(report, indent=2))
