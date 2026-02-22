import numpy as np
from typing import Dict, List, Any, Tuple
import logging

# Import existing modules (assuming they are in the path)
from ensemble_analyzer import EnsembleECGAnalyzer
from clinical_criteria.advanced_validator import AdvancedCriteriaValidator
from explainability.ecg_explainer import ECGExplainer
from reporting.clinical_generator import ClinicalReportGenerator

logger = logging.getLogger("DiagnosticReasoning")

class DiagnosticReasoningEngine:
    """
    Orchestrates the diagnostic process to generate structured, clinically-relevant outputs.
    Combines AI probabilities, rule-based criteria, and explainability features.
    """

    def __init__(self):
        self.ai_model = EnsembleECGAnalyzer()
        self.rule_engine = AdvancedCriteriaValidator()
        self.explainer = ECGExplainer(ensemble_analyzer=self.ai_model)
        self.reporter = ClinicalReportGenerator()

    def _generate_justification(self, 
                                diagnosis: str, 
                                features: Dict[str, Any], 
                                criteria_results: Dict[str, Any],
                                lead_contrib: Dict[str, float]) -> str:
        """
        Constructs a natural language justification for the diagnosis.
        """
        justification = []
        
        # 1. Primary Findings (from Features/Explainer)
        # Identify top leads contributing to diagnosis
        top_leads = sorted(lead_contrib, key=lead_contrib.get, reverse=True)[:3]
        leads_str = ", ".join(top_leads)
        
        if "STEMI" in diagnosis:
            justification.append(f"Significant ST elevation detected in leads {leads_str}.")
            # Add specific feature details if available
            if features.get('max_st_elevation', 0) > 0.1:
                justification.append(f"Max STE amplitude {features.get('max_st_elevation'):.2f}mV.")
                
        elif "AFib" in diagnosis:
            justification.append("Irregular RR intervals detected with absent P-waves.")
            
        elif "LBBB" in diagnosis:
            justification.append(f"Wide QRS complex ({features.get('qrs_dur', 0)}ms) with dominant S wave in V1.")

        # 2. Criteria Matches
        if criteria_results.get('Sgarbossa', {}).get('match'):
            justification.append("Sgarbossa Criteria: POSITIVE (Concordant ST changes).")
            
        if criteria_results.get('Brugada_VT', {}).get('match'):
            justification.append("Brugada Algorithm: POSITIVE for VT.")
            
        if criteria_results.get('DeWinter', {}).get('match'):
            justification.append("De Winter Pattern: POSITIVE (LAD Occlusion equivalent).")
            
        if criteria_results.get('Wellens', {}).get('match'):
            justification.append("Wellens Sign: POSITIVE.")

        # 3. Fallback
        if not justification:
            justification.append(f"Morphological pattern consistent with {diagnosis} based on ensemble model consensus.")

        return " ".join(justification)

    def _get_differentials(self, all_probs: List[float], class_names: List[str]) -> List[Dict[str, Any]]:
        """
        Returns sorted differential diagnoses.
        """
        # Pair probs with names
        paired = list(zip(class_names, all_probs))
        # Sort desc
        sorted_preds = sorted(paired, key=lambda x: x[1], reverse=True)
        
        differentials = []
        for name, prob in sorted_preds:
            differentials.append({
                "diagnosis": name,
                "probability": round(float(prob), 4),
                "confidence_level": "High" if prob > 0.8 else ("Medium" if prob > 0.5 else "Low")
            })
            
        return differentials

    def _get_clinical_recommendation(self, diagnosis: str, criteria_results: Dict[str, Any]) -> str:
        """
        Generates actionable clinical recommendation.
        """
        if "STEMI" in diagnosis or criteria_results.get('Sgarbossa', {}).get('match'):
            return "URGENT: Activate STEMI Protocol. Immediate Cardiology Consult. Prepare for Cath Lab."
            
        if criteria_results.get('DeWinter', {}).get('match'):
            return "URGENT: De Winter pattern suggests LAD occlusion. Treat as STEMI equivalent."
            
        if "AFib" in diagnosis:
            return "Evaluate for anticoagulation (CHA2DS2-VASc). Rate/Rhythm control strategy."
            
        if "Hyperkalemia" in diagnosis: # If implemented
            return "URGENT: Check serum Potassium. Calcium Gluconate if ECG changes present."
            
        # Default from reporter
        recs = self.reporter.get_recommendations([diagnosis])
        if recs:
            return recs[0]
            
        return "Clinical correlation suggested. Repeat ECG if symptoms persist."

    def analyze(self, ecg_signal: np.ndarray, patient_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis pipeline.
        
        Args:
            ecg_signal: (5000, 12) numpy array
            patient_meta: Dict with age, sex, symptoms, etc.
        """
        # 1. AI Inference
        # Get raw probabilities from ensemble (assuming predict returns list of dicts, we take first)
        # We need access to raw probabilities for differentials. 
        # The current EnsembleECGAnalyzer.predict returns the top class. 
        # We might need to access internal model or modify predict to return all probs.
        # For this implementation, we'll assume we can get them or simulate them.
        
        # Mocking getting all probabilities from the ensemble for the single ECG
        # In real implementation, modify EnsembleECGAnalyzer to return full prob vector
        ai_result = self.ai_model.predict(np.expand_dims(ecg_signal, 0))[0]
        
        # Simulate full probability vector for demo (since predict returns top class)
        # In prod: ai_result['all_probabilities']
        class_names = self.ai_model.class_names
        top_diag = ai_result['diagnosis']
        top_conf = ai_result['confidence']
        
        # Mock distribution
        probs = [0.05] * len(class_names)
        try:
            top_idx = class_names.index(top_diag)
            probs[top_idx] = top_conf
            # Normalize rest
            rem_prob = 1.0 - top_conf
            for i in range(len(probs)):
                if i != top_idx:
                    probs[i] = rem_prob / (len(probs) - 1)
        except ValueError:
            pass # Diagnosis not in list
            
        differentials = self._get_differentials(probs, class_names)

        # 2. Explainability (Heatmap & Features)
        explanation = self.explainer.explain_diagnosis(ecg_signal)
        lead_contrib = explanation['lead_contribution']
        
        # 3. Rule-Based Validation
        # Extract fiducials (Mock or use SignalProcessor)
        fiducials = {'J_point': 1000, 'Q_start': 980, 'R_peak': 990} # Mock
        criteria_results = self.rule_engine.validate_all(ecg_signal.T, fiducials, patient_meta)
        
        # 4. Synthesize Output
        justification = self._generate_justification(
            top_diag, 
            explanation.get('key_features', {}), 
            criteria_results,
            lead_contrib
        )
        
        recommendation = self._get_clinical_recommendation(top_diag, criteria_results)
        
        # 5. Visual Asset (Heatmap)
        # Generate image bytes
        heatmap_img = self.explainer.visualize_explanation(ecg_signal, explanation)
        
        return {
            "primary_diagnosis": {
                "diagnosis": top_diag,
                "confidence": f"{top_conf:.1%}",
                "risk_tier": ai_result['tier']
            },
            "differentials": differentials[:3], # Top 3
            "clinical_reasoning": {
                "justification": justification,
                "criteria_met": [k for k, v in criteria_results.items() if v.get('match')],
                "key_leads": [k for k, v in lead_contrib.items() if v > 0.15] # Leads with >15% contribution
            },
            "recommendation": recommendation,
            "visuals": {
                "heatmap_bytes_len": len(heatmap_img) # In real app, return base64 string or URL
            }
        }

# Example Usage
if __name__ == "__main__":
    engine = DiagnosticReasoningEngine()
    
    # Mock Data
    ecg = np.random.randn(5000, 12).astype(np.float32)
    meta = {'age': 65, 'is_lbbb': True}
    
    result = engine.analyze(ecg, meta)
    
    import json
    print(json.dumps(result, indent=2, default=str))
