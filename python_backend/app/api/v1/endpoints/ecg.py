from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid

from app.db.models import get_db, ECGRecord, Patient, User
from app.worker import analyze_ecg_task
from security.auth import AuthManager, TokenData # Reusing previous auth logic
from security.rbac import RoleChecker

router = APIRouter()

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
    
    # Check Celery status if still processing? 
    # For now, rely on DB status updated by worker (mocked in worker)
    
    return {
        "id": record.id,
        "status": record.status,
        "diagnosis": record.diagnosis_main,
        "confidence": record.confidence,
        "timestamp": record.timestamp
    }

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
        # Trigger feedback loop for retraining
        # feedback_system.submit(...)
        
    db.commit()
    return {"status": "reviewed"}
