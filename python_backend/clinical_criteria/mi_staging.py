import numpy as np
from typing import Dict, List, Any, Optional

class MIStagingClassifier:
    """
    Categorizes Myocardial Infarction (MI) evolution stages based on ECG morphology.
    
    Phases:
    - Hyperacute (0-2h): ST elevation + Tall/Peaked T waves
    - Acute (2-24h): ST elevation + Emerging Q waves + R wave loss
    - Subacute (24h - weeks): Resolving ST + Deep T inversion + Q waves
    - Old/Scar (> weeks): Isoelectric ST + Persistent Q waves
    """
    
    def __init__(self):
        # Thresholds (mV)
        self.STE_THRESHOLD = 0.1 # 1mm
        self.TALL_T_THRESHOLD = 1.0 # 10mm (Hyperacute T) or relative to R
        self.T_INV_THRESHOLD = -0.1 # 1mm inversion
        
    def classify_stage(self, ecg_features: Dict[str, Any], clinical_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Determines the MI stage.
        
        Args:
            ecg_features: Dict containing:
                - max_st_elevation (mV)
                - max_t_amplitude (mV)
                - min_t_amplitude (mV)
                - has_pathological_q (bool)
                - leads_with_ste (List[str])
            clinical_info: Optional dict with 'onset_hours'
            
        Returns:
            Dict with 'stage', 'confidence', 'description'
        """
        max_ste = ecg_features.get('max_st_elevation', 0.0)
        max_t = ecg_features.get('max_t_amplitude', 0.0)
        min_t = ecg_features.get('min_t_amplitude', 0.0)
        has_q = ecg_features.get('has_pathological_q', False)
        
        # Default
        stage = "Indeterminate"
        confidence = 0.0
        reasoning = []
        
        # 1. Hyperacute Phase
        # Criteria: Significant STE + Tall T + No Q (usually)
        if max_ste >= self.STE_THRESHOLD:
            if max_t >= self.TALL_T_THRESHOLD or (max_t > 0.5 and not has_q):
                stage = "Hyperacute"
                confidence = 0.9
                reasoning.append("Significant ST elevation with hyperacute T waves.")
                if not has_q:
                    reasoning.append("Absence of pathological Q waves.")
            
            # 2. Acute Phase
            # Criteria: Significant STE + Q waves appearing or T wave starting to invert
            elif has_q or min_t < 0:
                stage = "Acute"
                confidence = 0.85
                reasoning.append("ST elevation with developing Q waves or T wave changes.")
            else:
                # Fallback for just STE
                stage = "Acute" 
                confidence = 0.7
                reasoning.append("ST elevation present (evolving).")
                
        # 3. Subacute (Recent) Phase
        # Criteria: Resolving STE (less than acute but maybe not isoelectric) + Deep T Inversion + Q waves
        elif max_ste < self.STE_THRESHOLD and min_t <= self.T_INV_THRESHOLD:
            if has_q:
                stage = "Subacute"
                confidence = 0.9
                reasoning.append("Pathological Q waves with deep T wave inversion.")
                reasoning.append("ST segment returning to baseline.")
            else:
                stage = "Subacute (Ischemia)"
                confidence = 0.7
                reasoning.append("Deep T wave inversion without significant Q waves.")

        # 4. Old / Scar Phase
        # Criteria: Isoelectric ST + Persistent Q + T wave normalization (or persistent inversion)
        elif has_q and max_ste < self.STE_THRESHOLD:
            stage = "Old/Scar"
            confidence = 0.95
            reasoning.append("Pathological Q waves with isoelectric ST segment.")
            if min_t > self.T_INV_THRESHOLD:
                 reasoning.append("T waves normalized.")
            else:
                 reasoning.append("Persistent T wave inversion.")

        # 5. Normal / Non-MI
        elif not has_q and max_ste < self.STE_THRESHOLD and min_t > self.T_INV_THRESHOLD:
            stage = "Normal / Non-Specific"
            confidence = 0.9
            reasoning.append("No significant ST-T or Q wave abnormalities.")

        # Clinical Correlation (if available)
        if clinical_info and 'onset_hours' in clinical_info:
            hours = clinical_info['onset_hours']
            clinical_stage = self._get_clinical_stage(hours)
            
            if stage == clinical_stage:
                confidence = min(1.0, confidence + 0.1)
                reasoning.append(f"Matches clinical timeline ({hours}h).")
            elif stage != "Indeterminate" and clinical_stage != "Unknown":
                reasoning.append(f"Discordant with clinical timeline ({hours}h -> {clinical_stage}).")
                # Adjust stage if ECG is ambiguous but time is clear? 
                # Usually ECG morphology trumps time for "ECG Staging", but time trumps for "Clinical Staging".
                # We stick to ECG morphology for this classifier.

        return {
            "stage": stage,
            "confidence": confidence,
            "reasoning": "; ".join(reasoning)
        }

    def _get_clinical_stage(self, hours: float) -> str:
        if hours < 2: return "Hyperacute"
        if hours < 24: return "Acute"
        if hours < 24 * 21: return "Subacute" # < 3 weeks
        return "Old/Scar"

# Example
if __name__ == "__main__":
    classifier = MIStagingClassifier()
    
    # Case 1: Hyperacute
    feats1 = {
        'max_st_elevation': 0.3, # 3mm
        'max_t_amplitude': 1.2, # 12mm
        'min_t_amplitude': 0.0,
        'has_pathological_q': False
    }
    print("Case 1:", classifier.classify_stage(feats1))
    
    # Case 2: Old Inferior MI
    feats2 = {
        'max_st_elevation': 0.05,
        'max_t_amplitude': 0.3,
        'min_t_amplitude': -0.05,
        'has_pathological_q': True
    }
    print("Case 2:", classifier.classify_stage(feats2))
