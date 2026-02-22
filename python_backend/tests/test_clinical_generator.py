import pytest
import json
from reporting.clinical_generator import ClinicalReportGenerator

def test_diagnosis_mapping_pt():
    gen = ClinicalReportGenerator(language='pt')
    mapped = gen.diagnosis_to_icd10(["STEMI"])
    assert mapped[0]['icd10'] == "I21.3"
    assert mapped[0]['name'] == "Infarto Agudo do Miocárdio com Supra de ST (IAMCSST)"
    assert mapped[0]['severity'] == "High"

def test_recommendations_en():
    gen = ClinicalReportGenerator(language='en')
    recs = gen.get_recommendations(["Atrial Fibrillation"])
    assert "Rate vs Rhythm control." in recs
    assert len(recs) >= 1

def test_fhir_generation():
    gen = ClinicalReportGenerator()
    analysis = {
        "diagnosis": "NSTEMI",
        "metrics": {"hr": 80, "qtc": 420}
    }
    fhir = gen.generate_fhir_observation(analysis, "P999")
    
    assert fhir['resourceType'] == "Observation"
    assert fhir['code']['coding'][0]['code'] == "11524-6" # LOINC EKG
    assert fhir['subject']['reference'] == "Patient/P999"
    
    # Check components
    hr_comp = next(c for c in fhir['component'] if c['code']['coding'][0]['code'] == "8867-4")
    assert hr_comp['valueQuantity']['value'] == 80

def test_full_report_structure():
    gen = ClinicalReportGenerator(language='es')
    analysis = {
        "patient_id": "P55",
        "diagnosis": "LBBB",
        "metrics": {"qrs": 140}
    }
    patient = {"id": "P55", "name": "Maria"}
    
    report = gen.generate_report(analysis, patient)
    
    assert report['language'] == 'es'
    assert report['clinical_findings']['risk_level'] == "Medium"
    assert report['clinical_findings']['diagnoses'][0]['name'] == "Bloqueio de Rama Izquierda"
