import numpy as np
from typing import Dict, List, Any, Tuple

class HRVRiskAssessor:
    """
    Heart Rate Variability (HRV) Analysis for Cardiac Risk Stratification.
    Calculates Time-Domain and Non-Linear metrics to assess Autonomic Nervous System (ANS) function.
    """
    
    def __init__(self):
        pass

    def _get_rr_intervals(self, r_peaks: np.ndarray, fs: int = 500) -> np.ndarray:
        """
        Converts R-peak indices to RR intervals in milliseconds.
        """
        if len(r_peaks) < 2:
            return np.array([])
        
        # Difference between peaks
        rr_samples = np.diff(r_peaks)
        # Convert to ms
        rr_ms = (rr_samples / fs) * 1000
        return rr_ms

    def calculate_time_domain(self, rr_ms: np.ndarray) -> Dict[str, float]:
        """
        Calculates standard time-domain metrics: RMSSD, SDNN, pNN50.
        """
        if len(rr_ms) < 2:
            return {"RMSSD": 0, "SDNN": 0, "pNN50": 0, "MeanNN": 0}
            
        # Successive differences
        diff_rr = np.diff(rr_ms)
        
        # RMSSD: Root Mean Square of Successive Differences
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        
        # SDNN: Standard Deviation of NN intervals
        sdnn = np.std(rr_ms, ddof=1)
        
        # pNN50: Percentage of differences > 50ms
        nn50 = np.sum(np.abs(diff_rr) > 50)
        pnn50 = (nn50 / len(diff_rr)) * 100
        
        return {
            "RMSSD": round(rmssd, 2),
            "SDNN": round(sdnn, 2),
            "pNN50": round(pnn50, 2),
            "MeanNN": round(np.mean(rr_ms), 2)
        }

    def calculate_dfa_alpha1(self, rr_ms: np.ndarray) -> float:
        """
        Simplified Detrended Fluctuation Analysis (DFA) alpha-1 (short-term correlations).
        Full DFA requires more complex logic; this is a simplified slope estimation 
        of F(n) vs n for small scales (4-16 beats).
        """
        if len(rr_ms) < 20:
            return 1.0 # Insufficient data, assume healthy 1/f
            
        # Integrate the series (cum sum of deviation from mean)
        y = np.cumsum(rr_ms - np.mean(rr_ms))
        
        scales = [4, 8, 12, 16]
        fluctuations = []
        
        for scale in scales:
            # Split into windows
            n_windows = len(y) // scale
            if n_windows < 1: continue
            
            rms = []
            for i in range(n_windows):
                window = y[i*scale : (i+1)*scale]
                x = np.arange(scale)
                # Detrend (linear fit)
                coef = np.polyfit(x, window, 1)
                trend = np.polyval(coef, x)
                # RMS of residual
                rms.append(np.sqrt(np.mean((window - trend)**2)))
            
            fluctuations.append(np.mean(rms))
            
        if len(fluctuations) < 2:
            return 1.0
            
        # Slope of log-log plot
        log_scales = np.log10(scales[:len(fluctuations)])
        log_fluc = np.log10(fluctuations)
        
        alpha1 = np.polyfit(log_scales, log_fluc, 1)[0]
        return round(alpha1, 2)

    def assess_risk(self, r_peaks: np.ndarray, fs: int = 500) -> Dict[str, Any]:
        """
        Main assessment function.
        """
        rr_ms = self._get_rr_intervals(r_peaks, fs)
        
        # Filter outliers (simple artifact removal: 300ms < RR < 2000ms)
        rr_clean = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        
        if len(rr_clean) < 10:
            return {"error": "Insufficient valid RR intervals for HRV analysis."}
            
        time_metrics = self.calculate_time_domain(rr_clean)
        dfa_alpha1 = self.calculate_dfa_alpha1(rr_clean)
        
        # Risk Stratification
        rmssd = time_metrics['RMSSD']
        sdnn = time_metrics['SDNN']
        
        # Sudden Cardiac Death (SCD) Risk
        # Low HRV is a predictor
        scd_risk = "Low"
        scd_factors = []
        
        if sdnn < 50: # Compromised health
            scd_risk = "High"
            scd_factors.append("Severely reduced global variability (SDNN < 50ms)")
        elif sdnn < 100:
            scd_risk = "Moderate"
            
        if rmssd < 20:
            if scd_risk == "Low": scd_risk = "Moderate"
            scd_factors.append("Reduced vagal tone (RMSSD < 20ms)")
            
        # Arrhythmia Prognosis
        # DFA alpha1: ~1.0 is healthy. < 0.75 (random) or > 1.25 (smooth) is abnormal.
        # Reduced alpha1 often precedes AFib onset.
        arrhythmia_risk = "Low"
        if dfa_alpha1 < 0.75:
            arrhythmia_risk = "High"
            scd_factors.append(f"Fractal correlation breakdown (DFA a1 {dfa_alpha1})")
        elif dfa_alpha1 > 1.25:
            arrhythmia_risk = "Moderate"
            
        return {
            "metrics": {
                **time_metrics,
                "DFA_alpha1": dfa_alpha1
            },
            "risk_assessment": {
                "sudden_cardiac_death_risk": scd_risk,
                "arrhythmia_prognosis": arrhythmia_risk,
                "risk_factors": scd_factors
            },
            "interpretation": f"HRV analysis indicates {scd_risk} risk of SCD. ANS balance appears {'compromised' if scd_risk != 'Low' else 'normal'}."
        }

# Example Usage
if __name__ == "__main__":
    assessor = HRVRiskAssessor()
    
    # Mock R-peaks (indices)
    # 1. Healthy: ~60bpm (1000ms) with variability
    r_peaks_healthy = np.cumsum(np.random.normal(1000, 50, 100)).astype(int) # SD=50ms
    
    # 2. At Risk: ~60bpm (1000ms) with very low variability
    r_peaks_risk = np.cumsum(np.random.normal(1000, 5, 100)).astype(int) # SD=5ms
    
    print("Healthy Subject:")
    print(assessor.assess_risk(r_peaks_healthy, fs=1000))
    
    print("\nHigh Risk Subject:")
    print(assessor.assess_risk(r_peaks_risk, fs=1000))
