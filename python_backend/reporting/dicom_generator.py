import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom._storage_sopclass_uids import BasicTextSRStorage
import datetime
import os
from typing import Dict, Any, List

class DicomReportGenerator:
    """
    Generates DICOM Structured Reports (SR) for ECG Analysis.
    Compatible with PACS/RIS systems.
    """
    
    def __init__(self, institution_name: str = "CardioAI Hospital"):
        self.institution_name = institution_name

    def create_dicom_sr(self, 
                        patient_data: Dict[str, Any], 
                        analysis_results: Dict[str, Any], 
                        output_path: str = "report.dcm") -> str:
        """
        Creates a DICOM Basic Text SR object containing the analysis.
        """
        
        # 1. Create File Meta Information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = BasicTextSRStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        # 2. Create Dataset
        ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # 3. Patient / Study / Series Information
        ds.PatientName = patient_data.get('name', 'Anonymous')
        ds.PatientID = patient_data.get('mrn', 'UNKNOWN')
        ds.PatientBirthDate = patient_data.get('dob', '19000101')
        ds.PatientSex = patient_data.get('sex', 'O')
        
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = BasicTextSRStorage
        
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
        ds.AccessionNumber = patient_data.get('accession', '')
        ds.Modality = 'SR'
        ds.SeriesNumber = 1
        ds.InstanceNumber = 1
        
        ds.Manufacturer = "CardioAI"
        ds.InstitutionName = self.institution_name
        
        # 4. SR Specific Content
        # Value Type
        ds.ValueType = 'CONTAINER'
        ds.ConceptNameCodeSequence = self._create_code_seq("18748-4", "LOINC", "Diagnostic Imaging Report")
        ds.ContinuityOfContent = 'SEPARATE'
        
        # Content Sequence
        content_seq = []
        
        # A. Summary / Diagnosis
        diagnosis = analysis_results.get('primary_diagnosis', {}).get('diagnosis', 'Unknown')
        confidence = analysis_results.get('primary_diagnosis', {}).get('confidence', '0%')
        
        content_seq.append(self._create_text_item(
            "121071", "DCM", "Finding", 
            f"Primary Diagnosis: {diagnosis} (Confidence: {confidence})"
        ))
        
        # B. Measurements
        metrics = analysis_results.get('metrics', {})
        if metrics:
            measurements_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            content_seq.append(self._create_text_item(
                "121070", "DCM", "Findings", 
                f"Measurements:\n{measurements_text}"
            ))
            
        # C. Clinical Recommendation
        rec = analysis_results.get('recommendation', 'None')
        content_seq.append(self._create_text_item(
            "121075", "DCM", "Recommendation", 
            f"Clinical Action: {rec}"
        ))
        
        # D. Alerts
        alerts = analysis_results.get('alerts', [])
        if alerts:
            alert_text = "; ".join([a.get('diagnosis', 'Alert') for a in alerts])
            content_seq.append(self._create_text_item(
                "121073", "DCM", "Impression", 
                f"Critical Alerts: {alert_text}"
            ))

        ds.ContentSequence = content_seq
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        # Save
        ds.save_as(output_path)
        return output_path

    def _create_code_seq(self, code_value, coding_scheme, code_meaning):
        """Helper to create Code Sequence."""
        ds = Dataset()
        ds.CodeValue = code_value
        ds.CodingSchemeDesignator = coding_scheme
        ds.CodeMeaning = code_meaning
        return [ds]

    def _create_text_item(self, concept_code, scheme, meaning, text_value):
        """Helper to create a TEXT Content Item."""
        item = Dataset()
        item.RelationshipType = 'CONTAINS'
        item.ValueType = 'TEXT'
        item.ConceptNameCodeSequence = self._create_code_seq(concept_code, scheme, meaning)
        item.TextValue = text_value[:64000] # Limit length
        return item

# Example Usage
if __name__ == "__main__":
    generator = DicomReportGenerator()
    
    pat = {'name': 'Doe^John', 'mrn': '12345', 'sex': 'M'}
    res = {
        'primary_diagnosis': {'diagnosis': 'STEMI', 'confidence': '98%'},
        'metrics': {'HR': '85 bpm', 'QTc': '460 ms'},
        'recommendation': 'Activate Cath Lab',
        'alerts': [{'diagnosis': 'Hyperkalemia'}]
    }
    
    path = generator.create_dicom_sr(pat, res, "test_ecg_sr.dcm")
    print(f"DICOM SR saved to {path}")
