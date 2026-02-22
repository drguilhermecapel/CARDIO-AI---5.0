import pytest
import json
from reporting.report_generator import ECGReportGenerator
from reporting.digital_signature import DigitalSignature

def test_digital_signature():
    signer = DigitalSignature()
    data = "Critical Medical Report"
    sig = signer.sign_data(data)
    assert signer.verify_signature(data, sig) is True
    assert signer.verify_signature("Tampered Report", sig) is False

def test_report_generation():
    generator = ECGReportGenerator()
    
    patient = {"id": "P1", "name": "Test Patient", "sex": "Female", "dob": "1990-01-01"}
    analysis = {
        "diagnosis": "STEMI",
        "confidence": 0.99,
        "metrics": {"hr": 80, "qtc": 450},
        "device_id": "DEV1"
    }
    
    report = generator.create_report(patient, analysis)
    
    # Check JSON structure
    assert report['json']['findings']['diagnosis'] == "STEMI"
    assert report['json']['findings']['icd10_code'] == "I21.3"
    assert 'signature' in report['json']
    
    # Check HTML generation
    assert "<html>" in report['html']
    assert "STEMI" in report['html']
    
    # Check XML generation
    assert "ClinicalDocument" in report['xml_cda']
    assert "11524-6" in report['xml_cda'] # LOINC code

def test_comparison():
    generator = ECGReportGenerator()
    
    report1 = {
        "findings": {
            "diagnosis": "Normal Sinus Rhythm",
            "metrics": {"qtc": 400, "hr": 70}
        }
    }
    
    report2 = {
        "findings": {
            "diagnosis": "Atrial Fibrillation",
            "metrics": {"qtc": 440, "hr": 110}
        }
    }
    
    comp = generator.comparator.compare(report2, report1)
    assert comp['diagnosis_changed'] is True
    assert comp['significant_evolution'] is True
    assert comp['qtc_delta'] == 40
