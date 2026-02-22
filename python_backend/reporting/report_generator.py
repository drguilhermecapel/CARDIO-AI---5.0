from typing import Dict, Any, List
from datetime import datetime
from .exporters import HL7CDAExporter, PDFExporter, HTMLReporter
from .digital_signature import DigitalSignature
import json

class ComparisonAnalyzer:
    """
    Compares two ECG reports to detect changes.
    """
    def compare(self, current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
        curr_metrics = current.get('findings', {}).get('metrics', {})
        prev_metrics = previous.get('findings', {}).get('metrics', {})
        
        curr_diag = current.get('findings', {}).get('diagnosis', '')
        prev_diag = previous.get('findings', {}).get('diagnosis', '')
        
        changes = {
            "diagnosis_changed": curr_diag != prev_diag,
            "new_diagnosis": curr_diag if curr_diag != prev_diag else None,
            "qtc_delta": curr_metrics.get('qtc', 0) - prev_metrics.get('qtc', 0),
            "hr_delta": curr_metrics.get('hr', 0) - prev_metrics.get('hr', 0),
            "significant_evolution": False
        }
        
        # Logic for significance
        if changes['diagnosis_changed']:
            changes['significant_evolution'] = True
        if abs(changes['qtc_delta']) > 30: # >30ms change is significant
            changes['significant_evolution'] = True
            
        return changes

class ECGReportGenerator:
    """
    Orchestrates report generation in multiple formats.
    """
    def __init__(self, sw_version="5.0.0"):
        self.sw_version = sw_version
        self.signer = DigitalSignature()
        self.hl7 = HL7CDAExporter()
        self.pdf = PDFExporter()
        self.html = HTMLReporter()
        self.comparator = ComparisonAnalyzer()
        
        # ICD-10 Mapping (Simplified)
        self.icd10_map = {
            "Normal Sinus Rhythm": "R00.0",
            "Atrial Fibrillation": "I48.91",
            "STEMI": "I21.3",
            "NSTEMI": "I21.4",
            "PVCs": "I49.3",
            "LBBB": "I44.7",
            "RBBB": "I45.1"
        }

    def create_report(self, 
                      patient_data: Dict[str, Any], 
                      analysis_result: Dict[str, Any], 
                      previous_report: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generates full report package.
        """
        # 1. Prepare Data Structure
        diagnosis = analysis_result.get('diagnosis', 'Unknown')
        icd10 = self.icd10_map.get(diagnosis, "R94.31") # Abnormal ECG NOS
        
        report_data = {
            "patient": patient_data,
            "findings": {
                "diagnosis": diagnosis,
                "icd10_code": icd10,
                "confidence": analysis_result.get('confidence'),
                "metrics": analysis_result.get('metrics', {}),
                "is_abnormal": diagnosis != "Normal Sinus Rhythm"
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "sw_version": self.sw_version,
                "device_id": analysis_result.get('device_id', 'Unknown'),
                "model_version": analysis_result.get('model_version', 'v1.0')
            }
        }
        
        # 2. Comparison (if previous exists)
        if previous_report:
            comparison = self.comparator.compare(report_data, previous_report)
            report_data['comparison'] = comparison
            
        # 3. Generate Content for Signing
        # We sign the canonical JSON representation
        canonical_json = json.dumps(report_data, sort_keys=True)
        signature = self.signer.sign_data(canonical_json)
        report_data['signature'] = signature
        
        # 4. Export Formats
        exports = {
            "json": report_data,
            "xml_cda": self.hl7.generate(report_data),
            "html": self.html.generate(report_data),
            # PDF generation returns bytes, usually saved to file or base64
            # "pdf_bytes": self.pdf.generate(report_data) 
        }
        
        return exports

# Example Usage
if __name__ == "__main__":
    generator = ECGReportGenerator()
    
    patient = {"id": "P12345", "name": "John Doe", "sex": "Male", "dob": "1980-01-01", "age": 46}
    analysis = {
        "diagnosis": "Atrial Fibrillation",
        "confidence": 0.95,
        "metrics": {"hr": 110, "qtc": 440, "qrs": 90, "pr": 0, "axis": 45},
        "device_id": "ECG_001"
    }
    
    report = generator.create_report(patient, analysis)
    print("Report Generated.")
    print(f"Signature: {report['json']['signature'][:32]}...")
    print(f"ICD-10: {report['json']['findings']['icd10_code']}")
