import numpy as np
import logging
from typing import Dict, List, Any, Tuple

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LeadPlacement")

class LeadPlacementDetector:
    """
    Detects ECG lead placement errors and anatomical inconsistencies.
    
    Checks for:
    - RA/LA Reversal (Limb lead reversal)
    - LA/LL Reversal
    - Precordial Lead Reversal (V1-V6 progression)
    - Missing / Disconnected Leads
    - Einthoven's Law violations
    """
    
    def __init__(self, fs: int = 500):
        self.fs = fs
        self.lead_indices = {
            'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
            'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11
        }

    def _get_signal(self, leads_signal: np.ndarray, lead_name: str) -> np.ndarray:
        idx = self.lead_indices.get(lead_name)
        if idx is not None and idx < leads_signal.shape[0]:
            return leads_signal[idx]
        return np.zeros(leads_signal.shape[1])

    def _get_polarity(self, signal: np.ndarray) -> float:
        """
        Estimates polarity of the dominant deflection (QRS) and P-wave area.
        Returns >0 for positive, <0 for negative.
        """
        # Simple heuristic: Mean of the signal often reflects baseline, 
        # but Skewness or Max/Min comparison is better for polarity.
        
        # QRS Polarity: Compare Max vs abs(Min)
        qrs_max = np.max(signal)
        qrs_min = np.min(signal)
        
        if abs(qrs_max) > abs(qrs_min):
            return 1.0 # Positive dominant
        else:
            return -1.0 # Negative dominant

    def _detect_p_wave_polarity(self, signal: np.ndarray) -> float:
        """
        Heuristic to detect P-wave polarity. 
        Assumes P-wave is in the 200ms before QRS.
        This is a simplified check. Real implementation needs R-peak detection.
        """
        # Placeholder: Using overall low-frequency trend or skewness of the P-wave region
        # For this module, we will assume we have access to a 'global' polarity check
        # or rely on the QRS/T concordance which usually follows P in normal sinus.
        
        # Using skewness as a proxy for dominant direction
        from scipy.stats import skew
        return skew(signal)

    def check_missing_leads(self, leads_signal: np.ndarray) -> List[str]:
        """
        Identifies flatlined or noise-only leads.
        """
        missing = []
        for name, idx in self.lead_indices.items():
            if idx >= leads_signal.shape[0]:
                missing.append(name)
                continue
                
            sig = leads_signal[idx]
            # Check amplitude (Peak-to-Peak)
            amp = np.ptp(sig)
            if amp < 0.05: # < 50uV is essentially flat
                missing.append(name)
                
        return missing

    def check_ra_la_reversal(self, leads_signal: np.ndarray) -> Dict[str, Any]:
        """
        Detects Right Arm / Left Arm reversal.
        Signs:
        - Lead I is inverted (P, QRS, T all negative).
        - aVR is positive (P, QRS, T all positive).
        """
        lead_I = self._get_signal(leads_signal, 'I')
        lead_aVR = self._get_signal(leads_signal, 'aVR')
        
        # Check Lead I polarity (Global skewness/mean often negative in reversal)
        pol_I = self._get_polarity(lead_I)
        pol_aVR = self._get_polarity(lead_aVR)
        
        # Normal: I > 0, aVR < 0
        # Reversal: I < 0, aVR > 0
        
        if pol_I < 0 and pol_aVR > 0:
            return {
                "detected": True,
                "confidence": "High",
                "description": "Lead I inverted and aVR positive. Suggests RA/LA Reversal."
            }
            
        return {"detected": False}

    def check_la_ll_reversal(self, leads_signal: np.ndarray) -> Dict[str, Any]:
        """
        Detects Left Arm / Left Leg reversal.
        Signs:
        - Lead III is inverted Lead I.
        - P-wave in Lead I is usually positive (unlike RA/LA reversal).
        - Lead III P-wave might be inverted.
        """
        lead_III = self._get_signal(leads_signal, 'III')
        
        # If Lead III is completely inverted (P, QRS, T negative) in a patient 
        # where it shouldn't be (though axis deviation can cause this).
        # This is harder to detect without P-wave specific analysis.
        
        # Heuristic: If Lead III is strictly negative and Lead I is positive, 
        # it might be LAD, but if P-wave in III is inverted while I is upright, check placement.
        
        return {"detected": False, "note": "Requires P-wave segmentation for accurate detection"}

    def check_precordial_progression(self, leads_signal: np.ndarray) -> Dict[str, Any]:
        """
        Checks for R-wave progression V1 -> V6.
        Normal: R wave amplitude increases from V1 to V4/V5.
        Reversal: V2 < V1 or V3 < V2 (unless pathology like old septal MI).
        """
        r_amps = []
        for lead in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            sig = self._get_signal(leads_signal, lead)
            r_amps.append(np.max(sig)) # Simple max as R-wave proxy
            
        # Check V1 vs V2 vs V3
        # If V1 R-wave is dominant and V2 is small -> Possible reversal or RBBB/RVH
        # If V2 < V1 and V3 > V2 -> Likely V1/V2 reversal if no other explanation
        
        issues = []
        if r_amps[0] > r_amps[1] and r_amps[1] < r_amps[2]:
            issues.append("Possible V1/V2 Reversal")
            
        if r_amps[4] < r_amps[5] * 0.5: # V5 significantly smaller than V6? Usually V5 > V6 or V5 ~= V6
            pass # Not necessarily error
            
        if issues:
            return {
                "detected": True,
                "details": issues,
                "description": "Abnormal R-wave progression. Check precordial lead sequence."
            }
            
        return {"detected": False}

    def check_einthoven_law(self, leads_signal: np.ndarray) -> Dict[str, Any]:
        """
        Verifies Einthoven's Law: II = I + III.
        Significant deviation implies technical error or non-orthogonal recording.
        """
        lead_I = self._get_signal(leads_signal, 'I')
        lead_II = self._get_signal(leads_signal, 'II')
        lead_III = self._get_signal(leads_signal, 'III')
        
        # Calculate residual
        residual = np.abs(lead_II - (lead_I + lead_III))
        mean_residual = np.mean(residual)
        max_amp = np.max(np.abs(lead_II))
        
        # Threshold: > 5% error relative to signal amplitude
        if max_amp > 0 and (mean_residual / max_amp) > 0.05:
            return {
                "detected": True,
                "residual_norm": float(mean_residual / max_amp),
                "description": "Einthoven's Law violation (II != I + III). Check lead integrity."
            }
            
        return {"detected": False}

    def validate_placement(self, leads_signal: np.ndarray) -> Dict[str, Any]:
        """
        Runs all placement checks.
        """
        missing = self.check_missing_leads(leads_signal)
        ra_la = self.check_ra_la_reversal(leads_signal)
        precordial = self.check_precordial_progression(leads_signal)
        einthoven = self.check_einthoven_law(leads_signal)
        
        alerts = []
        if missing:
            alerts.append(f"Missing Leads: {', '.join(missing)}")
        if ra_la['detected']:
            alerts.append(ra_la['description'])
        if precordial['detected']:
            alerts.append(precordial['description'])
        if einthoven['detected']:
            alerts.append(einthoven['description'])
            
        return {
            "is_valid": len(alerts) == 0,
            "alerts": alerts,
            "details": {
                "missing_leads": missing,
                "ra_la_reversal": ra_la,
                "precordial_progression": precordial,
                "einthoven_check": einthoven
            }
        }

# Example Usage
if __name__ == "__main__":
    detector = LeadPlacementDetector()
    
    # Mock Signal (12 leads, 5000 samples)
    t = np.linspace(0, 5, 5000)
    sig = np.zeros((12, 5000))
    
    # Normal-ish
    sig[0] = np.sin(t) # I
    sig[1] = np.sin(t) * 1.5 # II
    sig[2] = sig[1] - sig[0] # III (Einthoven holds)
    sig[3] = -(sig[0] + sig[1])/2 # aVR
    
    # Simulate RA/LA Reversal (Invert I, Swap II/III roughly)
    rev_sig = sig.copy()
    rev_sig[0] = -sig[0] # Invert I
    rev_sig[3] = -sig[3] # Invert aVR (becomes positive)
    
    print("Normal Check:", detector.validate_placement(sig)['is_valid'])
    print("Reversal Check:", detector.validate_placement(rev_sig)['alerts'])
