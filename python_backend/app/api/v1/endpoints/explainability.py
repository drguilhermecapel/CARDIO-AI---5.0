from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from app.db.models import get_db, ECGRecord
from security.auth import AuthManager, User
from explainability.ecg_explainer import ECGExplainer
from ensemble_analyzer import EnsembleECGAnalyzer
import numpy as np
import json

router = APIRouter()

# Initialize Singletons (In prod, use dependency injection)
ensemble = EnsembleECGAnalyzer()
explainer = ECGExplainer(ensemble_analyzer=ensemble)

@router.get("/{id}/explain", response_model=dict)
async def explain_ecg(
    id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):
    """
    Get JSON explanation for ECG diagnosis.
    """
    record = db.query(ECGRecord).filter(ECGRecord.id == id).first()
    if not record:
        raise HTTPException(status_code=404, detail="ECG not found")
        
    # Load Raw Data (Mocking S3 load)
    # In real app: raw_data = load_from_s3(record.raw_data_path)
    # Mock:
    raw_data = np.random.randn(5000, 12).astype(np.float32)
    
    explanation = explainer.explain_diagnosis(raw_data)
    
    # Remove large arrays for JSON response
    explanation_json = explanation.copy()
    del explanation_json['saliency_map']
    
    return explanation_json

@router.get("/{id}/explain/image")
async def explain_ecg_image(
    id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):
    """
    Get Heatmap Image for ECG explanation.
    """
    record = db.query(ECGRecord).filter(ECGRecord.id == id).first()
    if not record:
        raise HTTPException(status_code=404, detail="ECG not found")
        
    # Mock Data
    raw_data = np.random.randn(5000, 12).astype(np.float32)
    
    explanation = explainer.explain_diagnosis(raw_data)
    img_bytes = explainer.visualize_explanation(raw_data, explanation)
    
    return Response(content=img_bytes, media_type="image/png")
