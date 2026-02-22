import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ClinicalDecisionSupport")

class ClinicalDecisionSupport:
    """
    Implements clinical protocols based on ECG findings and patient context.
    Handles STEMI, NSTEMI, and Mimics (Pericarditis, BER).
    """
    
    LEAD_GROUPS = {
        "Septal": ["V1", "V2"],
        "Anterior": ["V3", "V4"],
        "Lateral": ["I", "aVL", "V5", "V6"],
        "Inferior": ["II", "III", "aVF"],
        "Right": ["V1", "aVR"]
    }

    def __init__(self):
        pass

    def _check_contiguous_leads(self, leads_with_ste: List[str]) -> Tuple[bool, str]:
        """
        Checks if ST elevation is present in at least 2 contiguous leads.
        Returns (True/False, Territory Name).
        """
        leads_set = set(leads_with_ste)
        
        for name, group in self.LEAD_GROUPS.items():
            # Check intersection
            matches = [lead for lead in group if lead in leads_set]
            if len(matches) >= 2:
                return True, name
        
        return False, None

    def _check_pericarditis_features(self, ecg_features: Dict[str, Any]) -> bool:
        """
        Checks for features suggestive of Pericarditis:
        - Diffuse ST elevation (involving multiple territories).
        - PR depression (if available).
        """
        leads_ste = ecg_features.get('leads_with_ste', [])
        
        # Diffuse check: STE in both Anterior/Lateral AND Inferior
        has_ant_lat = any(l in leads_ste for l in self.LEAD_GROUPS["Anterior"] + self.LEAD_GROUPS["Lateral"])
        has_inf = any(l in leads_ste for l in self.LEAD_GROUPS["Inferior"])
        
        if len(leads_ste) >= 5 and has_ant_lat and has_inf:
            return True
            
        if ecg_features.get('has_pr_depression', False):
            return True
            
        return False

    def evaluate(self, ecg_features: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main decision logic.
        
        Args:
            ecg_features: Dict with 'leads_with_ste', 'leads_with_t_inv', 'max_st_elevation', 'has_pr_depression'
            patient_data: Dict with 'troponin_status' (Positive/Negative/Suspected), 'symptoms'
        """
        leads_ste = ecg_features.get('leads_with_ste', [])
        leads_t_inv = ecg_features.get('leads_with_t_inv', [])
        leads_std = ecg_features.get('leads_with_std', [])
        
        trop_status = patient_data.get('troponin_status', 'Unknown')
        symptoms = patient_data.get('symptoms', [])
        is_symptomatic = 'chest_pain' in symptoms or 'dyspnea' in symptoms
        
        # 1. STEMI Check
        is_contiguous, territory = self._check_contiguous_leads(leads_ste)
        
        if is_contiguous:
            # Check for Mimics (Pericarditis)
            if self._check_pericarditis_features(ecg_features):
                return {
                    "protocol": "Pericarditis / Mimic",
                    "action": "Evaluate for Pericarditis vs STEMI. Check PR depression, Spodick's sign.",
                    "target_time": "Urgent Echo",
                    "differential": ["Acute Pericarditis", "STEMI", "Benign Early Repolarization"],
                    "status": "WARNING"
                }
            
            # True STEMI
            return {
                "protocol": "STEMI Protocol",
                "action": "ACTIVATE CATH LAB. Immediate Reperfusion.",
                "target_time": "Door-to-Balloon < 90 min",
                "diagnosis": f"STEMI ({territory} Wall)",
                "status": "CRITICAL"
            }
            
        # 2. NSTEMI / UA Check
        # Significant T-inv or STD + (Troponin or Symptoms)
        has_ischemia_sign = (len(leads_t_inv) >= 2) or (len(leads_std) >= 2)
        
        if has_ischemia_sign:
            if trop_status == 'Positive' or trop_status == 'Suspected':
                return {
                    "protocol": "NSTEMI Protocol",
                    "action": "Dual Antiplatelet Therapy (DAPT), Anticoagulation, Admit to CCU.",
                    "target_time": "Early Invasive Strategy (< 24h)",
                    "diagnosis": "NSTEMI (likely)",
                    "status": "URGENT"
                }
            elif is_symptomatic:
                return {
                    "protocol": "Unstable Angina Protocol",
                    "action": "Serial Troponins, Monitor, Stress Test or Angio.",
                    "target_time": "Observation",
                    "diagnosis": "Unstable Angina / Ischemia",
                    "status": "HIGH_RISK"
                }

        # 3. Default / Low Risk
        return {
            "protocol": "Standard Care",
            "action": "Clinical Correlation. Repeat ECG if symptoms change.",
            "target_time": "N/A",
            "diagnosis": "Non-Diagnostic / Normal Variant",
            "status": "ROUTINE"
        }

# Example Usage
if __name__ == "__main__":
    cds = ClinicalDecisionSupport()
    
    # Case 1: Inferior STEMI
    ecg1 = {'leads_with_ste': ['II', 'III', 'aVF'], 'max_st_elevation': 0.2}
    print("Case 1:", cds.evaluate(ecg1, {}))
    
    # Case 2: Pericarditis
    ecg2 = {'leads_with_ste': ['I', 'II', 'V2', 'V3', 'V4', 'V5', 'V6'], 'has_pr_depression': True}
    print("Case 2:", cds.evaluate(ecg2, {}))
    
    # Case 3: NSTEMI
    ecg3 = {'leads_with_t_inv': ['V2', 'V3', 'V4'], 'leads_with_ste': []}
    pat3 = {'troponin_status': 'Positive'}
    print("Case 3:", cds.evaluate(ecg3, pat3))
