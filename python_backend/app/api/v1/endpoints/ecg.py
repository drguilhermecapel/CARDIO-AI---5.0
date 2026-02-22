from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Response
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid
import json

from app.db.models import get_db, ECGRecord, Patient, User
from app.worker import analyze_ecg_task
from security.auth import AuthManager, TokenData
from security.rbac import RoleChecker
from reporting.clinical_generator import ClinicalReportGenerator

router = APIRouter()
report_gen = ClinicalReportGenerator()

# --- Schemas ---
class ECGUpload(BaseModel):
    patient_id: str
    raw_data: List[float] # Simplified for demo
    sample_rate: int = 500
    device_id: str

class ECGResponse(BaseModel):
    id: str
    status: str
    diagnosis: Optional[str]
    confidence: Optional[float]
    timestamp: datetime

class ReviewRequest(BaseModel):
    diagnosis_correction: Optional[str]
    notes: str
    approved: bool

# --- Endpoints ---

@router.post("/analyze", response_model=ECGResponse)
async def analyze_ecg(
    upload: ECGUpload, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):
    """
    Submit ECG for analysis. Returns ID for polling.
    """
    # Create Record
    record_id = str(uuid.uuid4())
    db_record = ECGRecord(
        id=record_id,
        patient_id=upload.patient_id,
        status="PROCESSING",
        device_id=upload.device_id,
        sample_rate=upload.sample_rate,
        raw_data_path="s3://bucket/..." # Mock
    )
    db.add(db_record)
    db.commit()
    
    # Trigger Celery Task
    task = analyze_ecg_task.delay(record_id, {"data": upload.raw_data})
    
    return {
        "id": record_id,
        "status": "PROCESSING",
        "timestamp": datetime.utcnow(),
        "diagnosis": None,
        "confidence": None
    }

@router.get("/{id}/result", response_model=ECGResponse)
async def get_result(
    id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):
    record = db.query(ECGRecord).filter(ECGRecord.id == id).first()
    if not record:
        raise HTTPException(status_code=404, detail="ECG not found")
    
    return {
        "id": record.id,
        "status": record.status,
        "diagnosis": record.diagnosis_main,
        "confidence": record.confidence,
        "timestamp": record.timestamp
    }

@router.get("/{id}/report/pdf")
async def get_pdf_report(
    id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):
    """
    Download clinical PDF report.
    """
    record = db.query(ECGRecord).filter(ECGRecord.id == id).first()
    if not record or not record.report_json:
        raise HTTPException(status_code=404, detail="Report not available")
        
    # Mock Patient Data (In real app, fetch from Patient table)
    patient_data = {
        "id": record.patient_id,
        "name": "John Doe", # Mock
        "dob": "1980-01-01"
    }
    
    # Generate Report
    # We use the stored JSON report from the analysis
    analysis_result = record.report_json
    
    # Generate full structure
    full_report = report_gen.generate_report(analysis_result, patient_data)
    
    # Render PDF
    pdf_bytes = report_gen.export_to_pdf(full_report)
    
    return Response(content=pdf_bytes, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=ECG_Report_{id}.pdf"
    })

@router.get("/{id}/report/fhir")
async def get_fhir_observation(
    id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):
    """
    Get FHIR R4 Observation resource.
    """
    record = db.query(ECGRecord).filter(ECGRecord.id == id).first()
    if not record or not record.report_json:
        raise HTTPException(status_code=404, detail="Report not available")
        
    analysis_result = record.report_json
    fhir_resource = report_gen.generate_fhir_observation(analysis_result, record.patient_id)
    
    return fhir_resource

@router.get("/history/{patient_id}", response_model=List[ECGResponse])
async def get_history(
    patient_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):
    records = db.query(ECGRecord).filter(ECGRecord.patient_id == patient_id).order_by(ECGRecord.timestamp.desc()).all()
    return [
        {
            "id": r.id,
            "status": r.status,
            "diagnosis": r.diagnosis_main,
            "confidence": r.confidence,
            "timestamp": r.timestamp
        } for r in records
    ]

@router.post("/compare")
async def compare_ecgs(
    id_1: str, 
    id_2: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):
    r1 = db.query(ECGRecord).filter(ECGRecord.id == id_1).first()
    r2 = db.query(ECGRecord).filter(ECGRecord.id == id_2).first()
    
    if not r1 or not r2:
        raise HTTPException(status_code=404, detail="One or both records not found")
        
    # Mock Comparison Logic
    return {
        "comparison": {
            "rhythm_change": r1.diagnosis_main != r2.diagnosis_main,
            "qt_change_ms": 10, # Mock
            "new_ischemia": False
        }
    }

@router.post("/{id}/review", dependencies=[Depends(RoleChecker(["cardiologist", "admin"]))])
async def review_ecg(
    id: str, 
    review: ReviewRequest, 
    db: Session = Depends(get_db),
    current_user: tuple = Depends(AuthManager.get_current_user)
):
    user, _ = current_user
    record = db.query(ECGRecord).filter(ECGRecord.id == id).first()
    if not record:
        raise HTTPException(status_code=404, detail="ECG not found")
        
    record.is_reviewed = True
    record.reviewer_id = 1 # Mock user ID
    record.review_notes = review.notes
    record.reviewed_at = datetime.utcnow()
    
    if review.diagnosis_correction:
        record.diagnosis_main = review.diagnosis_correction
        
    db.commit()
    return {"status": "reviewed"}
