import logging
from typing import Dict, Any, Optional
import numpy as np

# Import all modules
from quality.quality_assessor import QualityAssessor
from quality.lead_placement import LeadPlacementDetector
from reasoning.diagnostic_engine import DiagnosticReasoningEngine
from clinical_criteria.bayesian_assessor import BayesianRiskAssessor
from clinical_criteria.mi_staging import MIStagingClassifier
from clinical_criteria.decision_support import ClinicalDecisionSupport
from analysis.temporal_comparator import TemporalComparator
from audit.audit_logger import AuditLogger

logger = logging.getLogger("CardioAIService")

class CardioAIIntegrationService:
    """
    Main Orchestrator for the CardioAI System.
    Integrates Quality, Diagnosis, Risk, Staging, Protocols, and Reporting.
    """
    
    def __init__(self, db_session=None):
        self.quality = QualityAssessor()
        self.leads = LeadPlacementDetector()
        self.diagnostic = DiagnosticReasoningEngine()
        self.bayesian = BayesianRiskAssessor()
        self.staging = MIStagingClassifier()
        self.protocols = ClinicalDecisionSupport()
        self.temporal = TemporalComparator()
        self.audit = AuditLogger(db_session) if db_session else None

    def process_ecg(self, 
                    ecg_signal: np.ndarray, 
                    patient_meta: Dict[str, Any], 
                    previous_ecg_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Full processing pipeline.
        """
        logger.info(f"Processing ECG for Patient {patient_meta.get('id', 'Unknown')}")
        
        # 1. Signal Quality & Lead Placement
        quality_report = self.quality.assess_quality(ecg_signal)
        lead_report = self.leads.validate_placement(ecg_signal)
        
        # Fail fast if unusable
        if quality_report['reliability'] == "Unusable":
            return {
                "status": "REJECTED",
                "error": "ECG signal quality is unusable.",
                "quality_report": quality_report
            }
            
        if not lead_report['is_valid']:
            # We might still process but with warnings
            logger.warning("Lead placement issues detected.")

        # 2. Diagnostic Inference (AI + Rules)
        # Assuming diagnostic engine extracts features internally or we do it here
        # For this integration, diagnostic engine does the heavy lifting
        diag_result = self.diagnostic.analyze(ecg_signal, patient_meta)
        
        # Extract features from diagnostic result for other modules
        # (In a real system, FeatureExtraction would be a separate shared step)
        # Mocking extracted features based on diagnosis for flow demonstration
        extracted_features = {
            'diagnosis': diag_result['primary_diagnosis']['diagnosis'],
            'max_st_elevation': 0.2 if "STEMI" in diag_result['primary_diagnosis']['diagnosis'] else 0.0,
            'leads_with_ste': diag_result['clinical_reasoning'].get('key_leads', []),
            'is_lbbb': "LBBB" in diag_result['primary_diagnosis']['diagnosis'],
            'qtc': 450 # Mock
        }
        
        # 3. Bayesian Risk Assessment
        risk_assessment = self.bayesian.assess_patient(patient_meta, {
            'diagnosis': extracted_features['diagnosis'],
            'has_st_deviation': extracted_features['max_st_elevation'] > 0.1
        })
        
        # 4. MI Staging (if relevant)
        staging_result = {}
        if "MI" in extracted_features['diagnosis'] or "STEMI" in extracted_features['diagnosis']:
            staging_result = self.staging.classify_stage(extracted_features, patient_meta)
            
        # 5. Temporal Comparison
        temporal_result = {}
        if previous_ecg_features:
            temporal_result = self.temporal.compare_records(extracted_features, previous_ecg_features)
            
        # 6. Clinical Decision Support (Protocols)
        protocol_result = self.protocols.evaluate(extracted_features, {
            'troponin_status': patient_meta.get('labs', {}).get('troponin_status', 'Unknown'),
            'symptoms': patient_meta.get('symptoms', [])
        })
        
        # 7. Audit Logging
        if self.audit:
            self.audit.log_analysis_event(
                record_id=patient_meta.get('record_id', 'new'),
                model_version="5.0.0",
                input_data={"patient": patient_meta.get('id')},
                result={"diagnosis": extracted_features['diagnosis']}
            )
            
        # 8. Final Report Assembly
        final_report = {
            "status": "COMPLETED",
            "patient_id": patient_meta.get('id'),
            "timestamp": "Now",
            "quality_check": {
                "score": quality_report['overall_score'],
                "alerts": quality_report['issues_detected'] + lead_report['alerts']
            },
            "diagnosis": diag_result['primary_diagnosis'],
            "differentials": diag_result['differentials'],
            "risk_profile": {
                "bayesian": risk_assessment['bayesian_analysis'],
                "scores": risk_assessment['risk_scores']
            },
            "staging": staging_result,
            "temporal_analysis": temporal_result,
            "clinical_protocol": protocol_result,
            "recommendation": protocol_result['action']
        }
        
        return final_report

# Example
if __name__ == "__main__":
    service = CardioAIIntegrationService()
    
    # Mock Input
    ecg = np.random.randn(12, 5000)
    meta = {
        'id': 'P123',
        'age': 60,
        'sex': 'Male',
        'symptoms': ['chest_pain'],
        'labs': {'troponin_status': 'Positive'}
    }
    prev_feats = {'is_lbbb': False, 'max_st_elevation': 0.0}
    
    report = service.process_ecg(ecg, meta, prev_feats)
    import json
    print(json.dumps(report, indent=2, default=str))
