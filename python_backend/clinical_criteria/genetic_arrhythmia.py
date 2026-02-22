import numpy as np
from typing import Dict, List, Any, Optional

class GeneticArrhythmiaDetector:
    """
    Detects genetic and structural arrhythmia syndromes.
    Includes: WPW, Brugada, Long/Short QT, Early Repolarization, CPVT markers.
    """
    
    def __init__(self):
        pass

    def check_wpw(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wolff-Parkinson-White (WPW) Syndrome.
        Criteria:
        - Short PR interval (< 120ms).
        - Wide QRS (> 110-120ms).
        - Delta wave (slurred upstroke of QRS) - represented by 'has_delta_wave' feature.
        """
        pr_int = features.get('pr_interval', 160)
        qrs_dur = features.get('qrs_dur', 90)
        has_delta = features.get('has_delta_wave', False)
        
        score = 0
        details = []
        
        if pr_int < 120:
            score += 1
            details.append(f"Short PR interval ({int(pr_int)}ms)")
            
        if qrs_dur > 110:
            score += 1
            details.append(f"Wide QRS ({int(qrs_dur)}ms)")
            
        if has_delta:
            score += 2
            details.append("Delta wave detected")
            
        # Diagnosis
        match = False
        if score >= 3: # Classic pattern
            match = True
        elif score >= 2 and has_delta: # Delta + one other
            match = True
            
        return {
            "match": match,
            "diagnosis": "Wolff-Parkinson-White (WPW) Pattern",
            "details": details,
            "recommendation": "Electrophysiology consult. Avoid AV nodal blockers if AFib present."
        }

    def check_brugada_detailed(self, lead_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detailed Brugada Syndrome Classification (V1-V3).
        Type 1: Coved STE >= 2mm, T-inv.
        Type 2: Saddleback STE >= 2mm, T positive/biphasic.
        Type 3: Saddleback/Coved < 2mm.
        """
        types_found = []
        leads_involved = []
        
        for lead in ['V1', 'V2', 'V3']:
            lf = lead_features.get(lead, {})
            ste = lf.get('st_elev', 0.0) # mV
            t_amp = lf.get('t_amp', 0.0)
            morph = lf.get('st_morphology', 'normal') # 'coved', 'saddleback'
            
            if ste >= 0.2: # >= 2mm
                if morph == 'coved' or (t_amp < 0): # Type 1 equivalent
                    types_found.append("Type 1")
                    leads_involved.append(lead)
                elif morph == 'saddleback':
                    types_found.append("Type 2")
                    leads_involved.append(lead)
            elif ste >= 0.1: # Type 3
                if morph in ['coved', 'saddleback']:
                    types_found.append("Type 3")
                    leads_involved.append(lead)

        if not types_found:
            return {"match": False}
            
        # Prioritize Type 1
        final_type = "Type 1" if "Type 1" in types_found else ("Type 2" if "Type 2" in types_found else "Type 3")
        
        return {
            "match": True,
            "diagnosis": f"Brugada Syndrome ({final_type})",
            "details": [f"{final_type} pattern in {', '.join(leads_involved)}"],
            "recommendation": "Arrhythmia specialist referral. Genetic testing. Avoid fever/drugs."
        }

    def check_long_qt_syndrome(self, features: Dict[str, Any], sex: str = 'Male') -> Dict[str, Any]:
        """
        Long QT Syndrome (LQTS).
        """
        qtc = features.get('qtc', 400)
        t_morph = features.get('t_morphology', 'normal') # 'broad', 'notched', 'late_onset'
        
        limit = 450 if sex == 'Male' else 460
        score = 0
        details = []
        
        # Schwartz Score components (Simplified)
        if qtc >= 480:
            score += 3
            details.append(f"QTc >= 480ms ({int(qtc)}ms)")
        elif qtc >= 460:
            score += 2
            details.append(f"QTc 460-479ms ({int(qtc)}ms)")
        elif qtc >= 450 and sex == 'Male':
            score += 1
            details.append(f"QTc 450-459ms (Male)")
            
        # T wave morphology
        if 'notched' in t_morph and features.get('leads_with_notched_t', []):
            score += 1
            details.append("Notched T waves (suggests LQT2)")
        if 'broad' in t_morph:
            score += 1
            details.append("Broad-based T waves (suggests LQT1)")
            
        # Bradycardia for age?
        
        prob = "Low"
        if score >= 3.5: prob = "High"
        elif score >= 1.5: prob = "Intermediate"
        
        return {
            "match": score >= 1.5,
            "diagnosis": "Long QT Syndrome",
            "probability": prob,
            "score": score,
            "details": details,
            "recommendation": "Genetic testing, Beta-blockers, Avoid QT prolonging meds."
        }

    def check_short_qt_syndrome(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Short QT Syndrome (SQTS).
        QTc < 340ms (High probability).
        QTc < 360ms + Symptoms/Family History.
        """
        qtc = features.get('qtc', 400)
        
        if qtc < 340:
            return {
                "match": True,
                "diagnosis": "Short QT Syndrome",
                "details": [f"QTc < 340ms ({int(qtc)}ms)"],
                "recommendation": "ICD evaluation. Quinidine may be considered."
            }
        elif qtc < 360:
            return {
                "match": True,
                "diagnosis": "Short QT Pattern (Borderline)",
                "details": [f"QTc < 360ms ({int(qtc)}ms)"],
                "recommendation": "Clinical correlation required."
            }
            
        return {"match": False}

    def check_early_repolarization(self, lead_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Early Repolarization Pattern (ERP).
        - J-point elevation >= 1mm in >= 2 contiguous leads (Inferior/Lateral).
        - Notching or slurring of terminal QRS.
        """
        # Check Inferior (II, III, aVF) and Lateral (I, aVL, V4-V6)
        groups = {
            "Inferior": ['II', 'III', 'aVF'],
            "Lateral": ['I', 'aVL', 'V4', 'V5', 'V6']
        }
        
        leads_with_j_elev = []
        for lead, feats in lead_features.items():
            j_elev = feats.get('j_point_elev', 0.0)
            has_notch = feats.get('has_qrs_notch', False)
            has_slur = feats.get('has_qrs_slur', False)
            
            if j_elev >= 0.1 and (has_notch or has_slur):
                leads_with_j_elev.append(lead)
                
        match = False
        territory = []
        
        for name, leads in groups.items():
            count = sum(1 for l in leads if l in leads_with_j_elev)
            if count >= 2:
                match = True
                territory.append(name)
                
        if match:
            return {
                "match": True,
                "diagnosis": "Early Repolarization Pattern",
                "details": [f"J-point elevation with notching in {', '.join(territory)} leads"],
                "recommendation": "Benign in most cases. Correlate with syncope/family history (Malignant ERP)."
            }
            
        return {"match": False}

    def check_cpvt_markers(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Catecholaminergic Polymorphic VT (CPVT).
        Resting ECG is often normal.
        Markers: Sinus bradycardia for age, Prominent U waves.
        """
        hr = features.get('hr', 70)
        age = features.get('age', 30)
        has_u = features.get('has_u_wave', False)
        
        # Bradycardia check
        is_brady = False
        if age < 10 and hr < 70: is_brady = True
        elif age < 20 and hr < 60: is_brady = True
        elif hr < 50: is_brady = True
        
        if is_brady or has_u:
            return {
                "match": True, # Low specificity
                "diagnosis": "CPVT Markers (Non-diagnostic)",
                "details": ["Resting bradycardia" if is_brady else "", "Prominent U waves" if has_u else ""],
                "recommendation": "CPVT is a stress-induced diagnosis. Exercise Stress Test or Holter required if symptomatic."
            }
            
        return {"match": False}

    def run_all(self, features: Dict[str, Any], lead_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = [
            self.check_wpw(features),
            self.check_brugada_detailed(lead_features),
            self.check_long_qt_syndrome(features, features.get('sex', 'Male')),
            self.check_short_qt_syndrome(features),
            self.check_early_repolarization(lead_features),
            self.check_cpvt_markers(features)
        ]
        return [r for r in results if r['match']]

# Example
if __name__ == "__main__":
    detector = GeneticArrhythmiaDetector()
    
    # WPW Case
    feats_wpw = {'pr_interval': 100, 'qrs_dur': 130, 'has_delta_wave': True}
    print("WPW:", detector.check_wpw(feats_wpw))
    
    # Brugada Type 1
    lead_feats_brug = {
        'V1': {'st_elev': 0.3, 'st_morphology': 'coved', 't_amp': -0.2},
        'V2': {'st_elev': 0.25, 'st_morphology': 'coved'}
    }
    print("Brugada:", detector.check_brugada_detailed(lead_feats_brug))
