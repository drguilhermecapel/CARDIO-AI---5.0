import numpy as np
from typing import Dict, List, Any, Optional

class SpecialPatternsDetector:
    """
    Detects specific clinical patterns and metabolic abnormalities affecting ECG interpretation.
    Includes: Hyperkalemia, Hypokalemia, Pulmonary Embolism, Takotsubo, Brugada.
    """
    
    def __init__(self):
        pass

    def check_hyperkalemia(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects signs of Hyperkalemia:
        - Tall, peaked (tented) T waves (narrow base).
        - Flattened P waves.
        - Wide QRS (severe).
        """
        # Criteria: Max T amplitude > 1.0 mV (10mm) or T/R ratio > 0.75 in V2-V4
        max_t = features.get('max_t_amplitude', 0.0)
        t_r_ratio = features.get('max_t_r_ratio', 0.0)
        qrs_dur = features.get('qrs_dur', 90)
        
        score = 0
        details = []
        
        if max_t > 1.0:
            score += 2
            details.append("Giant T waves (>1.0mV)")
        elif max_t > 0.6 and t_r_ratio > 0.75:
            score += 2
            details.append("Peaked T waves relative to R wave")
            
        if qrs_dur > 120:
            score += 1
            details.append("Wide QRS complex")
            
        return {
            "match": score >= 2,
            "diagnosis": "Hyperkalemia",
            "severity": "Severe" if qrs_dur > 140 else "Moderate",
            "details": details,
            "recommendation": "URGENT: Check Serum Potassium. Evaluate for immediate treatment."
        }

    def check_hypokalemia(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects signs of Hypokalemia:
        - Prominent U waves (> 1mm or > 25% of T wave).
        - T wave flattening/inversion.
        - ST depression.
        - Long QU interval.
        """
        has_u_wave = features.get('has_u_wave', False)
        u_amp = features.get('u_wave_amplitude', 0.0)
        t_amp = features.get('max_t_amplitude', 0.5)
        st_dep = features.get('max_st_depression', 0.0)
        
        score = 0
        details = []
        
        if has_u_wave:
            if u_amp > 0.1 or (t_amp > 0 and u_amp/t_amp > 0.25):
                score += 3
                details.append("Prominent U waves detected")
        
        if t_amp < 0.1 and t_amp > -0.1: # Flat T
            score += 1
            details.append("Flattened T waves")
            
        if st_dep > 0.05:
            score += 1
            details.append("ST depression")

        return {
            "match": score >= 3,
            "diagnosis": "Hypokalemia",
            "details": details,
            "recommendation": "Check Serum Potassium and Magnesium."
        }

    def check_pulmonary_embolism(self, features: Dict[str, Any], lead_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects S1Q3T3 Pattern for Pulmonary Embolism.
        - Lead I: Deep S wave (> 1.5mm)
        - Lead III: Q wave (> 1.5mm) + Inverted T wave
        """
        s_i = abs(lead_features.get('I', {}).get('s_amp', 0.0))
        q_iii = abs(lead_features.get('III', {}).get('q_amp', 0.0))
        t_iii = lead_features.get('III', {}).get('t_amp', 0.0)
        
        # Thresholds (mV)
        THRESHOLD = 0.15 # 1.5mm
        
        match = False
        details = []
        
        if s_i > THRESHOLD:
            details.append("Deep S in Lead I")
            if q_iii > THRESHOLD:
                details.append("Q wave in Lead III")
                if t_iii < -0.05: # Inverted
                    details.append("Inverted T in Lead III")
                    match = True
        
        # Tachycardia check
        hr = features.get('hr', 70)
        if hr > 100:
            details.append("Sinus Tachycardia")
            
        return {
            "match": match,
            "diagnosis": "Pulmonary Embolism (S1Q3T3)",
            "details": details,
            "recommendation": "Evaluate for PE (D-dimer, CT Angio). Correlate with symptoms (dyspnea, chest pain)."
        }

    def check_takotsubo(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects Takotsubo Cardiomyopathy (Apical Ballooning).
        - Deep, symmetric T wave inversion in precordial leads (V1-V6).
        - Significant QT prolongation.
        - Absence of reciprocal ST depression (unlike STEMI).
        """
        min_t = features.get('min_t_amplitude', 0.0)
        qtc = features.get('qtc', 400)
        has_reciprocal = features.get('has_reciprocal_changes', False)
        
        match = False
        details = []
        
        if min_t < -0.5: # Deep inversion > 5mm
            details.append("Deep T wave inversion")
            if qtc > 500: # Significant prolongation
                details.append("Marked QTc prolongation (>500ms)")
                if not has_reciprocal:
                    details.append("No reciprocal ST changes")
                    match = True
                    
        return {
            "match": match,
            "diagnosis": "Takotsubo Cardiomyopathy",
            "details": details,
            "recommendation": "Consider Echocardiogram to rule out Apical Ballooning vs ACS."
        }

    def check_brugada(self, features: Dict[str, Any], lead_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Brugada Syndrome (Type 1).
        - Coved-type ST elevation > 2mm in V1-V3.
        - Negative T wave.
        """
        # Check V1, V2
        match = False
        details = []
        
        for lead in ['V1', 'V2']:
            ste = lead_features.get(lead, {}).get('st_elev', 0.0)
            t_amp = lead_features.get(lead, {}).get('t_amp', 0.0)
            
            if ste > 0.2: # > 2mm
                if t_amp < 0: # Inverted T
                    details.append(f"Coved STE > 2mm with inverted T in {lead}")
                    match = True
        
        return {
            "match": match,
            "diagnosis": "Brugada Syndrome (Type 1)",
            "details": details,
            "recommendation": "URGENT: Arrhythmia specialist consult. Avoid fever and sodium channel blockers."
        }

    def run_all_checks(self, features: Dict[str, Any], lead_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Runs all special pattern checks and returns positive findings.
        """
        alerts = []
        
        checks = [
            self.check_hyperkalemia(features),
            self.check_hypokalemia(features),
            self.check_pulmonary_embolism(features, lead_features),
            self.check_takotsubo(features),
            self.check_brugada(features, lead_features)
        ]
        
        for res in checks:
            if res['match']:
                alerts.append(res)
                
        return alerts

# Example Usage
if __name__ == "__main__":
    detector = SpecialPatternsDetector()
    
    # Mock Features for Hyperkalemia
    feats_hyperk = {'max_t_amplitude': 1.2, 'qrs_dur': 100}
    print("HyperK:", detector.check_hyperkalemia(feats_hyperk))
    
    # Mock Features for PE
    feats_pe = {'hr': 110}
    lead_feats_pe = {
        'I': {'s_amp': -0.2}, # Deep S
        'III': {'q_amp': -0.2, 't_amp': -0.1} # Q + Inv T
    }
    print("PE:", detector.check_pulmonary_embolism(feats_pe, lead_feats_pe))
