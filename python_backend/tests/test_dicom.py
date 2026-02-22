import pytest
import os
import pydicom
from reporting.dicom_generator import DicomReportGenerator

@pytest.fixture
def generator():
    return DicomReportGenerator()

def test_dicom_sr_creation(generator, tmpdir):
    output_path = str(tmpdir.join("report.dcm"))
    
    pat = {'name': 'Test^Patient', 'mrn': 'TEST001', 'sex': 'F'}
    res = {
        'primary_diagnosis': {'diagnosis': 'Normal Sinus Rhythm', 'confidence': '99%'},
        'metrics': {'HR': '60'},
        'recommendation': 'Routine Follow-up'
    }
    
    path = generator.create_dicom_sr(pat, res, output_path)
    
    assert os.path.exists(path)
    
    # Read back
    ds = pydicom.dcmread(path)
    assert ds.PatientName == 'Test^Patient'
    assert ds.Modality == 'SR'
    assert ds.ValueType == 'CONTAINER'
    
    # Check content
    content = ds.ContentSequence
    assert len(content) >= 3
    assert "Normal Sinus Rhythm" in content[0].TextValue
