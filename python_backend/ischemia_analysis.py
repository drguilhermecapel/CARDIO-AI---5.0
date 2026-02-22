import numpy as np
from typing import Dict, List, Optional, Tuple

class IschemiaAnalyzer:
    """
    Advanced Ischemia Detection Engine following ESC/ACC Guidelines.
    Detects STEMI, NSTEMI, and Pathological Q-waves with demographic adjustment.
    """
    
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    TERRITORIES = {
        'Septal': ['V1', 'V2'],
        'Anterior': ['V3', 'V4'],
        'Lateral': ['I', 'aVL', 'V5', 'V6'],
        'Inferior': ['II', 'III', 'aVF'],
        'Right Ventricular': ['V1', 'V4R'], # V4R usually not present in standard 12-lead
        'Posterior': ['V7', 'V8', 'V9'] # Usually inferred from V1-V3 reciprocal changes
    }

    def __init__(self, age: int = 50, sex: str = 'Male', fs: int = 500):
        self.age = age
        self.sex = sex
        self.fs = fs
        
        # Define STEMI Thresholds (in mV) based on ESC Guidelines
        self.thresholds = self._get_stemi_thresholds()

    def _get_stemi_thresholds(self) -> Dict[str, float]:
        """
        Returns ST-elevation thresholds (mV) for V2-V3 and other leads.
        """
        # Standard leads: 0.1 mV (1mm)
        thresholds = {lead: 0.1 for lead in self.LEAD_NAMES}
        
        # V2-V3 Special Criteria
        if self.sex == 'Female':
            v2_v3_thresh = 0.15 # 1.5mm
        else:
            if self.age < 40:
                v2_v3_thresh = 0.25 # 2.5mm
            else:
                v2_v3_thresh = 0.20 # 2.0mm
        
        thresholds['V2'] = v2_v3_thresh
        thresholds['V3'] = v2_v3_thresh
        
        return thresholds

    def _measure_amplitude(self, signal: np.ndarray, idx: int, baseline: float) -> float:
        """Measure amplitude in mV relative to baseline."""
        if idx is None or np.isnan(idx) or idx >= len(signal) or idx < 0:
            return 0.0
        return signal[int(idx)] - baseline

    def analyze(self, leads_signal: np.ndarray, fiducials: Dict[str, List[int]]) -> Dict:
        """
        Main analysis pipeline.
        leads_signal: shape (12, samples)
        fiducials: dict of 'R', 'S', 'J', 'iso' indices per beat or average beat.
                   Assuming we process an "average beat" or representative beat for each lead.
                   For this implementation, we expect a dict where keys are lead names 
                   and values are dicts of fiducial indices for that lead.
        """
        # Note: If fiducials is just one set for all leads (sync), we use that.
        # If it's per lead, we iterate.
        # Let's assume input is (12, samples) and we have one set of global fiducials 
        # (e.g., from Lead II) or we re-detect per lead. 
        # For robustness, ischemia should be measured on the specific lead's morphology.
        # Here we assume 'fiducials' contains 'J_point' and 'Isoelectric_point' indices.
        
        findings = {
            "st_elevation": [],
            "st_depression": [],
            "t_wave_inversion": [],
            "pathological_q": [],
            "diagnosis": "Normal",
            "probability": 0.0,
            "affected_territory": []
        }

        # Mocking fiducial extraction if not provided per lead
        # In a real scenario, we'd use the WaveDetector on each lead.
        
        # Let's iterate leads
        st_deviations = {} # Lead -> mV
        
        for i, lead_name in enumerate(self.LEAD_NAMES):
            signal = leads_signal[i]
            
            # 1. Determine Baseline (PR segment or TP segment)
            # Simplified: Use the provided isoelectric point or start of signal
            iso_idx = fiducials.get('iso', 0)
            baseline = signal[iso_idx] if iso_idx < len(signal) else 0.0
            
            # 2. Determine J-point
            # If not provided, approximate from S or R
            j_idx = fiducials.get('J')
            if j_idx is None:
                # Fallback: R + 40ms
                r_idx = fiducials.get('R', [0])[0] # Assuming list of R peaks, take first/avg
                j_idx = r_idx + int(0.04 * self.fs)
            
            # 3. Measure ST at J, J+60ms, J+80ms
            st_j = self._measure_amplitude(signal, j_idx, baseline)
            st_60 = self._measure_amplitude(signal, j_idx + int(0.06 * self.fs), baseline)
            
            st_deviations[lead_name] = st_j
            
            # --- ST Elevation ---
            if st_j >= self.thresholds[lead_name]:
                findings["st_elevation"].append({
                    "lead": lead_name,
                    "magnitude_mv": round(st_j, 3),
                    "threshold_mv": self.thresholds[lead_name]
                })

            # --- ST Depression ---
            # Criteria: Horizontal/Downsloping >= 0.05 mV (0.5mm)
            # Check slope between J and J+60
            slope = st_60 - st_j
            if st_j <= -0.05:
                # If slope is negative (downsloping) or flat (horizontal)
                # Upsloping depression is often normal (J-point depression) unless De Winter
                if slope <= 0.02: # Tolerance for horizontal
                    findings["st_depression"].append({
                        "lead": lead_name,
                        "magnitude_mv": round(st_j, 3),
                        "type": "Downsloping" if slope < -0.01 else "Horizontal"
                    })

            # --- T Wave Inversion ---
            # Measure T peak. Assuming we have T index.
            t_idx = fiducials.get('T')
            if t_idx:
                t_amp = self._measure_amplitude(signal, t_idx, baseline)
                if t_amp <= -0.1: # -1mm
                     findings["t_wave_inversion"].append({
                        "lead": lead_name,
                        "magnitude_mv": round(t_amp, 3)
                    })
            
            # --- Pathological Q Wave ---
            # Q dur > 40ms, Amp > 25% R
            q_idx = fiducials.get('Q')
            r_idx = fiducials.get('R', [0])[0]
            if q_idx and r_idx:
                q_dur = (r_idx - q_idx) / self.fs # Rough approx of Q duration
                q_amp = abs(self._measure_amplitude(signal, q_idx, baseline))
                r_amp = self._measure_amplitude(signal, r_idx, baseline)
                
                if q_dur > 0.04 or (r_amp > 0 and q_amp / r_amp > 0.25):
                     findings["pathological_q"].append({
                        "lead": lead_name,
                        "duration_ms": round(q_dur * 1000, 1)
                    })

        # --- Diagnosis Logic (Contiguity Check) ---
        ste_leads = [f['lead'] for f in findings['st_elevation']]
        std_leads = [f['lead'] for f in findings['st_depression']]
        
        territories_involved = set()
        
        # Check Contiguity for STEMI
        is_stemi = False
        for territory, leads in self.TERRITORIES.items():
            # Count leads with STE in this territory
            count = sum(1 for lead in leads if lead in ste_leads)
            if count >= 2:
                territories_involved.add(territory)
                is_stemi = True

        if is_stemi:
            findings["diagnosis"] = "STEMI"
            findings["affected_territory"] = list(territories_involved)
            findings["probability"] = 0.98
            findings["confidence"] = "High"
        elif len(std_leads) >= 2:
            findings["diagnosis"] = "Ischemia / NSTEMI"
            findings["probability"] = 0.85
            findings["confidence"] = "Medium"
        elif findings["t_wave_inversion"]:
            findings["diagnosis"] = "T-Wave Abnormality (Possible Ischemia)"
            findings["probability"] = 0.60
            findings["confidence"] = "Low"

        return findings

# Example Usage
if __name__ == "__main__":
    # Mock Signal
    leads_signal = np.random.normal(0, 0.05, (12, 5000))
    # Inject STEMI in V3, V4 (Anterior)
    # Lead indices: V3=8, V4=9 (0-indexed: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)
    # Wait, indices are 0..11. 
    # I=0, II=1, III=2, aVR=3, aVL=4, aVF=5, V1=6, V2=7, V3=8, V4=9, V5=10, V6=11
    
    # Add ST elevation to V3 (idx 8) and V4 (idx 9)
    # R peak at 1000. J point at 1020.
    leads_signal[8, 1020:1100] += 0.3 # 3mm STE
    leads_signal[9, 1020:1100] += 0.3
    
    analyzer = IschemiaAnalyzer(age=55, sex='Male')
    
    # Mock Fiducials (Single beat)
    fiducials = {
        'iso': 900,
        'Q': 980,
        'R': [1000],
        'J': 1020,
        'T': 1200
    }
    
    result = analyzer.analyze(leads_signal, fiducials)
    import json
    print(json.dumps(result, indent=2))
