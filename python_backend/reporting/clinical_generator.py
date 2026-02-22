import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import jsonschema
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

from .clinical_codes import DIAGNOSIS_MAP

class ClinicalReportGenerator:
    """
    Generates clinical reports with ICD-10 mapping, FHIR compliance, and multi-language support.
    """
    
    def __init__(self, language: str = "en"):
        self.language = language if language in ['en', 'pt', 'es'] else 'en'
        self.schema = {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "diagnosis": {"type": "string"},
                "metrics": {"type": "object"},
                "confidence": {"type": "number"}
            },
            "required": ["patient_id", "diagnosis"]
        }

    def diagnosis_to_icd10(self, diagnosis_list: List[str]) -> List[Dict[str, str]]:
        """
        Maps internal diagnosis strings to standardized codes.
        """
        mapped = []
        for diag in diagnosis_list:
            entry = DIAGNOSIS_MAP.get(diag)
            if entry:
                mapped.append({
                    "name": entry['translations'][self.language],
                    "icd10": entry['icd10'],
                    "snomed": entry['snomed'],
                    "severity": entry['severity']
                })
            else:
                mapped.append({
                    "name": diag,
                    "icd10": "R94.31", # Abnormal ECG NOS
                    "snomed": "Unknown",
                    "severity": "Unknown"
                })
        return mapped

    def get_recommendations(self, diagnosis_list: List[str]) -> List[str]:
        """
        Retrieves clinical recommendations based on diagnosis and language.
        """
        recs = []
        for diag in diagnosis_list:
            entry = DIAGNOSIS_MAP.get(diag)
            if entry:
                recs.extend(entry['recommendations'][self.language])
        return list(set(recs)) # Deduplicate

    def generate_fhir_observation(self, analysis: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """
        Generates a FHIR R4 Observation resource for the ECG analysis.
        """
        diagnosis = analysis.get('diagnosis', 'Normal Sinus Rhythm')
        code_entry = DIAGNOSIS_MAP.get(diagnosis)
        
        snomed_code = code_entry['snomed'] if code_entry else "Unknown"
        display_text = code_entry['translations']['en'] if code_entry else diagnosis
        
        observation = {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "procedure",
                    "display": "Procedure"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "11524-6",
                    "display": "EKG study"
                }]
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": snomed_code,
                    "display": display_text
                }],
                "text": display_text
            },
            "component": []
        }
        
        # Add Metrics as components
        metrics = analysis.get('metrics', {})
        metric_codes = {
            "hr": ("8867-4", "Heart rate", "/min"),
            "qtc": ("44913-8", "QTc interval", "ms"),
            "qrs": ("8633-0", "QRS duration", "ms"),
            "pr": ("8625-6", "P-R interval", "ms")
        }
        
        for key, (loinc, name, unit) in metric_codes.items():
            if key in metrics:
                observation['component'].append({
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": loinc,
                            "display": name
                        }]
                    },
                    "valueQuantity": {
                        "value": metrics[key],
                        "unit": unit,
                        "system": "http://unitsofmeasure.org",
                        "code": unit
                    }
                })
                
        return observation

    def generate_report(self, ecg_analysis: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates the full structured clinical report.
        """
        # Validate Input (Soft validation)
        try:
            jsonschema.validate(instance=ecg_analysis, schema=self.schema)
        except jsonschema.ValidationError as e:
            print(f"Schema Validation Warning: {e.message}")

        diagnosis_list = [ecg_analysis.get('diagnosis', 'Normal Sinus Rhythm')]
        mapped_diagnoses = self.diagnosis_to_icd10(diagnosis_list)
        recommendations = self.get_recommendations(diagnosis_list)
        
        # Determine overall severity
        severities = [d['severity'] for d in mapped_diagnoses]
        if "High" in severities:
            risk_level = "High"
        elif "Medium" in severities:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        report = {
            "report_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "language": self.language,
            "patient": patient_data,
            "clinical_findings": {
                "diagnoses": mapped_diagnoses,
                "risk_level": risk_level,
                "confidence_tier": ecg_analysis.get('confidence_tier', 1),
                "metrics": ecg_analysis.get('metrics', {}),
                "evidence_text": ecg_analysis.get('evidence_text', "No specific evidence details provided.")
            },
            "recommendations": recommendations,
            "audit_trail": {
                "model_version": ecg_analysis.get('model_version', '5.0'),
                "generated_by": "CardioAI Nexus Clinical Engine",
                "reviewer": ecg_analysis.get('reviewer', 'Auto-Generated')
            }
        }
        
        return report

    def export_to_pdf(self, report: Dict[str, Any]) -> bytes:
        """
        Renders the report dictionary to a PDF file (bytes).
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Header
        title = "Relatório Clínico de ECG" if self.language == 'pt' else "ECG Clinical Report"
        story.append(Paragraph(f"<b>{title}</b>", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Patient Info
        pat = report['patient']
        p_text = f"<b>Patient:</b> {pat.get('name', 'N/A')} | <b>ID:</b> {pat.get('id', 'N/A')} | <b>DOB:</b> {pat.get('dob', 'N/A')}"
        story.append(Paragraph(p_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Diagnosis Section
        findings = report['clinical_findings']
        diag_color = colors.red if findings['risk_level'] == "High" else colors.black
        
        diag_style = ParagraphStyle('Diag', parent=styles['Heading2'], textColor=diag_color)
        story.append(Paragraph("Diagnosis / Diagnóstico", styles['Heading3']))
        
        for d in findings['diagnoses']:
            d_text = f"{d['name']} (ICD-10: {d['icd10']})"
            story.append(Paragraph(d_text, diag_style))
            
        story.append(Spacer(1, 12))
        
        # Metrics
        metrics = findings['metrics']
        data = [['Metric', 'Value']]
        for k, v in metrics.items():
            data.append([k.upper(), str(v)])
            
        t = Table(data, colWidths=[100, 100])
        t.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
        
        # Recommendations
        story.append(Paragraph("Recommendations / Conduta", styles['Heading3']))
        for rec in report['recommendations']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            
        doc.build(story)
        return buffer.getvalue()

# Example Usage
if __name__ == "__main__":
    # Test PT Generator
    gen_pt = ClinicalReportGenerator(language='pt')
    
    analysis = {
        "patient_id": "P123",
        "diagnosis": "STEMI",
        "confidence": 0.98,
        "metrics": {"hr": 95, "qtc": 460},
        "evidence_text": "Supra de ST em V1-V4."
    }
    
    patient = {"id": "P123", "name": "João Silva", "dob": "1960-05-20"}
    
    report = gen_pt.generate_report(analysis, patient)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    fhir = gen_pt.generate_fhir_observation(analysis, "P123")
    print("\nFHIR Resource:")
    print(json.dumps(fhir, indent=2))
