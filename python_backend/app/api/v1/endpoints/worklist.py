from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.db.models import get_db, ECGRecord
from security.auth import AuthManager
from security.rbac import RoleChecker

router = APIRouter()

@router.get("/", response_model=List[dict])
async def get_worklist(
    priority_only: bool = False,
    db: Session = Depends(get_db),
    current_user: tuple = Depends(AuthManager.get_current_user) # Requires auth
):
    """
    Returns list of unreviewed ECGs for cardiologists.
    """
    query = db.query(ECGRecord).filter(ECGRecord.is_reviewed == False)
    
    if priority_only:
        # Filter by confidence < 0.8 or Emergency status
        # Mock logic
        pass
        
    records = query.order_by(ECGRecord.timestamp.asc()).limit(50).all()
    
    return [
        {
            "id": r.id,
            "patient_id": r.patient_id,
            "timestamp": r.timestamp,
            "ai_diagnosis": r.diagnosis_main,
            "confidence": r.confidence,
            "status": "PENDING_REVIEW"
        } for r in records
    ]
