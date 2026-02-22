import numpy as np
from typing import Dict, List, Any, Optional

class SpecialPatternsDetector:
    """
    Detects specific ECG syndromes and patterns:
    - WPW (Wolff-Parkinson-White)
    - Brugada Syndrome (Type 1 & 2)
    - LQTS / SQTS (Long/Short QT Syndromes)
    - Early Repolarization
    - Epsilon Waves (ARVC)
    - Osborn Waves (Hypothermia)
    """
    
    def __init__(self, fs: int = 500):
        self.fs = fs

    def calculate_qtc(self, qt_ms: float, rr_ms: float) -> Dict[str, float]:
        """
        Calculates QTc using multiple formulas.
        """
        if rr_ms <= 0: return {}
        
        rr_sec = rr_ms / 1000.0
        
        # Bazett: QT / sqrt(RR)
        bazett = qt_ms / np.sqrt(rr_sec)
        
        # Fridericia: QT / cbrt(RR)
        fridericia = qt_ms / (rr_sec ** (1/3))
        
        # Framingham: QT + 0.154 * (1 - RR) * 1000
        framingham = qt_ms + 154 * (1 - rr_sec)
        
        # Hodges: QT + 1.75 * (HR - 60)
        hr = 60 / rr_sec
        hodges = qt_ms + 1.75 * (hr - 60)
        
        return {
            "Bazett": round(bazett, 1),
            "Fridericia": round(fridericia, 1),
            "Framingham": round(framingham, 1),
            "Hodges": round(hodges, 1)
        }

    def detect_wpw(self, 
                   pr_interval_ms: float, 
                   qrs_duration_ms: float, 
                   leads_signal: np.ndarray, 
                   fiducials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects WPW Pattern.
        Criteria: PR < 120ms, QRS > 110ms, Delta Wave.
        """
        # 1. Basic Interval Check
        if pr_interval_ms > 120:
            return {"detected": False, "reason": "PR interval normal (>120ms)"}
            
        if qrs_duration_ms < 110:
            return {"detected": False, "reason": "QRS duration normal (<110ms)"}

        # 2. Delta Wave Detection (Slurring of QRS upstroke)
        # We look at the first 20-30ms of the QRS complex.
        # If the slope is significantly lower than the rest of the upstroke, it's a delta wave.
        
        delta_wave_score = 0
        leads_with_delta = []
        
        # Check V1-V6
        lead_indices = [6, 7, 8, 9, 10, 11] # V1-V6
        
        q_start = fiducials.get('Q_start') # Global or per lead
        r_peak = fiducials.get('R_peak')
        
        if q_start is None or r_peak is None:
             return {"detected": False, "reason": "Fiducials missing"}

        for idx in lead_indices:
            signal = leads_signal[idx]
            
            # Define windows
            # Initial 20ms of QRS
            start_idx = int(q_start)
            mid_idx = start_idx + int(0.020 * self.fs) # 20ms
            end_idx = int(r_peak)
            
            if mid_idx >= end_idx or start_idx >= len(signal): continue
            
            # Calculate slopes
            slope_initial = (signal[mid_idx] - signal[start_idx]) / (mid_idx - start_idx)
            slope_rest = (signal[end_idx] - signal[mid_idx]) / (end_idx - mid_idx)
            
            # If initial slope is much flatter than the rest (e.g., < 50%)
            # Note: Slopes can be negative (Q wave). We look at absolute change or specific morphology.
            # Simplified: Delta wave usually positive in V4-V6.
            
            if abs(slope_initial) < 0.5 * abs(slope_rest):
                delta_wave_score += 1
                leads_with_delta.append(idx)

        is_wpw = delta_wave_score >= 2
        
        return {
            "detected": is_wpw,
            "confidence": 0.95 if is_wpw else 0.0,
            "delta_wave_leads": leads_with_delta,
            "criteria": "PR < 120ms + Wide QRS + Delta Wave"
        }

    def detect_brugada(self, leads_signal: np.ndarray, fiducials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects Brugada Syndrome (Type 1 & 2) in V1/V2.
        """
        # V1 is index 6, V2 is index 7
        v1 = leads_signal[6]
        v2 = leads_signal[7]
        
        j_point = fiducials.get('J_point')
        if j_point is None: return {"detected": False}
        
        j_idx = int(j_point)
        t_peak_idx = fiducials.get('T_peak')
        
        # Measure Amplitude at J and J+40ms
        # Baseline assumed 0 or passed in
        
        def check_lead(signal):
            st_j = signal[j_idx]
            st_40 = signal[j_idx + int(0.040 * self.fs)]
            
            # Type 1: Coved ST elevation > 2mm (0.2mV) + inverted T
            # "Coved": Convex shape. ST_40 < ST_J? Or specific curvature.
            # Simplified: High J point, negative T wave.
            
            is_type1 = False
            if st_j > 0.2:
                # Check T wave polarity
                if t_peak_idx:
                    t_amp = signal[int(t_peak_idx)]
                    if t_amp < 0: # Inverted T
                        is_type1 = True
            
            # Type 2: Saddleback > 2mm
            # J > 2mm, ST min > 1mm, Positive T
            is_type2 = False
            if st_j > 0.2:
                if t_peak_idx:
                    t_amp = signal[int(t_peak_idx)]
                    if t_amp > 0:
                        is_type2 = True
            
            return is_type1, is_type2

        v1_t1, v1_t2 = check_lead(v1)
        v2_t1, v2_t2 = check_lead(v2)
        
        if v1_t1 or v2_t1:
            return {"detected": True, "type": "Type 1 (Coved)", "leads": ["V1" if v1_t1 else "V2"], "risk": "High"}
        elif v1_t2 or v2_t2:
            return {"detected": True, "type": "Type 2 (Saddleback)", "leads": ["V1" if v1_t2 else "V2"], "risk": "Moderate"}
            
        return {"detected": False}

    def detect_lqts(self, qt_ms: float, rr_ms: float, sex: str) -> Dict[str, Any]:
        """
        Detects Long QT Syndrome.
        """
        qtc = self.calculate_qtc(qt_ms, rr_ms)
        val = qtc.get('Fridericia', 0) # Fridericia is preferred for extremes
        
        limit = 460 if sex == 'Female' else 450
        
        if val > 500:
            return {"detected": True, "type": "Severe LQTS", "qtc": val, "risk": "Very High (TdP)"}
        elif val > limit:
            return {"detected": True, "type": "Borderline/Long QT", "qtc": val, "risk": "Moderate"}
            
        return {"detected": False, "qtc": val}

    def detect_sqts(self, qt_ms: float, rr_ms: float) -> Dict[str, Any]:
        """
        Detects Short QT Syndrome.
        """
        qtc = self.calculate_qtc(qt_ms, rr_ms)
        val = qtc.get('Fridericia', 0)
        
        if val < 340:
             return {"detected": True, "qtc": val, "risk": "High (AFib/VF)"}
        elif val < 360:
             return {"detected": True, "qtc": val, "risk": "Moderate"}
             
        return {"detected": False}

    def detect_early_repol(self, leads_signal: np.ndarray, fiducials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects Early Repolarization Pattern (J-point elevation + notching).
        """
        j_point = fiducials.get('J_point')
        if j_point is None: return {"detected": False}
        
        # Check Inferior (II, III, aVF) and Lateral (I, aVL, V4-V6)
        # Indices: II=1, III=2, aVF=5, I=0, aVL=4, V4=9, V5=10, V6=11
        target_leads = [1, 2, 5, 9, 10, 11]
        
        erp_leads = []
        
        for idx in target_leads:
            sig = leads_signal[idx]
            st_j = sig[int(j_point)]
            
            # Criteria: J-point elevation > 0.1mV (1mm)
            if st_j > 0.1:
                # Check for notching/slurring (simplified)
                # Notching: small dip before ST segment rises?
                erp_leads.append(idx)
                
        if len(erp_leads) >= 2:
            return {"detected": True, "leads": erp_leads, "pattern": "J-point elevation"}
            
        return {"detected": False}

    def detect_epsilon(self, leads_signal: np.ndarray, fiducials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects Epsilon Waves (ARVC) in V1-V3.
        Small wiggles at the end of QRS / start of ST.
        """
        # Very hard to detect without high-res signal processing.
        # Placeholder for logic: High frequency energy in terminal QRS.
        return {"detected": False, "note": "Requires high-resolution signal analysis"}

    def detect_osborn(self, leads_signal: np.ndarray, fiducials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects Osborn (J) Waves.
        Prominent positive deflection at J-point, usually in hypothermia.
        """
        # Similar to ERP but usually larger and in precordial leads V2-V5
        j_point = fiducials.get('J_point')
        if j_point is None: return {"detected": False}
        
        # Check V3, V4, V5
        indices = [8, 9, 10]
        osborn_score = 0
        
        for idx in indices:
            sig = leads_signal[idx]
            st_j = sig[int(j_point)]
            
            # Osborn wave is a "hump". 
            # Check if J point is a local peak?
            if st_j > 0.2: # > 2mm
                osborn_score += 1
                
        if osborn_score >= 2:
            return {"detected": True, "risk": "Hypothermia / Hypercalcemia"}
            
        return {"detected": False}

    def analyze_all(self, 
                    leads_signal: np.ndarray, 
                    fiducials: Dict[str, Any], 
                    metrics: Dict[str, float],
                    patient_sex: str = 'Male') -> Dict[str, Any]:
        """
        Runs all detectors.
        """
        pr = metrics.get('PR_int_ms', 160)
        qrs = metrics.get('QRS_dur_ms', 90)
        qt = metrics.get('QT_int_ms', 400)
        rr = metrics.get('RR_int_ms', 800) # 75 bpm
        
        results = {
            "WPW": self.detect_wpw(pr, qrs, leads_signal, fiducials),
            "Brugada": self.detect_brugada(leads_signal, fiducials),
            "LQTS": self.detect_lqts(qt, rr, patient_sex),
            "SQTS": self.detect_sqts(qt, rr),
            "EarlyRepol": self.detect_early_repol(leads_signal, fiducials),
            "Osborn": self.detect_osborn(leads_signal, fiducials)
        }
        
        return results

# Example Usage
if __name__ == "__main__":
    detector = SpecialPatternsDetector()
    
    # Mock Signal (12 leads, 5000 samples)
    leads_signal = np.random.normal(0, 0.05, (12, 5000))
    
    # Mock Fiducials
    fiducials = {
        'Q_start': 980,
        'R_peak': 1000,
        'J_point': 1020,
        'T_peak': 1150
    }
    
    # Mock Metrics
    metrics = {
        'PR_int_ms': 100, # Short PR
        'QRS_dur_ms': 130, # Wide QRS
        'QT_int_ms': 400,
        'RR_int_ms': 800
    }
    
    # Inject Delta Wave in V4 (idx 9)
    # Slope upstroke
    leads_signal[9, 980:1000] = np.linspace(0, 0.5, 20) # Slow rise
    leads_signal[9, 1000:1020] = np.linspace(0.5, 1.0, 20) # Fast rise (R peak)
    
    results = detector.analyze_all(leads_signal, fiducials, metrics)
    import json
    print(json.dumps(results, indent=2))
