import os
import sys
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
from datetime import datetime
import logging
import json
import httpx

# Add parent directory to path to import sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.artifact_detection import LightweightArtifactDetector
from utils.xai_viz import IntegratedGradients

# Setup Audit Logger
logging.basicConfig(level=logging.INFO)
audit_logger = logging.getLogger("clinical_audit")
# Ensure logs are formatted as JSON for easy parsing
audit_handler = logging.StreamHandler(sys.stdout)
audit_handler.setFormatter(logging.Formatter('%(message)s'))
audit_logger.handlers = [audit_handler]
audit_logger.propagate = False

app = FastAPI(title="Cardio AI Comprehensive Endpoint")

# Global variables
model = None
explainer = None
artifact_detector = None

# Configuration
TRIAGE_THRESHOLDS = {
    "MI": float(os.environ.get("THRESH_MI", 0.6)),      # Critical
    "AFIB": float(os.environ.get("THRESH_AFIB", 0.8)),  # Urgent
    "PVC": float(os.environ.get("THRESH_PVC", 0.9))     # Routine/Urgent
}
WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")

class ECGRequest(BaseModel):
    signal: List[List[float]] # Shape: (5000, 12)
    mc_samples: int = 20
    explain: bool = False
    patient_id: Optional[str] = "UNKNOWN"

class TriageResult(BaseModel):
    status: str # "ROUTINE", "URGENT", "CRITICAL"
    reason: str
    action_required: bool

class FeedbackRequest(BaseModel):
    prediction_id: str
    patient_id: str
    model_version: str
    predicted_label: str
    corrected_label: str
    cardiologist_id: str
    comments: Optional[str] = None

class PredictionResponse(BaseModel):
    diagnosis: Dict[str, float]
    uncertainty: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]
    quality_metrics: Dict[str, float]
    triage: TriageResult
    explanations: Optional[Dict[str, Any]]
    metadata: Dict[str, str]
    prediction_id: str # Unique ID for feedback linking

def load_resources():
    global model, explainer, artifact_detector
    model_path = os.environ.get("MODEL_PATH", "models/hybrid/final_model")
    print(f"Loading model from {model_path}...")
    
    # Check if model exists, if not generate dummy for demo
    if not os.path.exists(model_path):
        print("Model not found. Generating dummy model for demonstration...")
        try:
            from scripts.setup_dummy_model import create_dummy_model
            create_dummy_model(model_path)
        except Exception as e:
            print(f"Failed to generate dummy model: {e}")

    try:
        model = tf.keras.models.load_model(model_path)
        explainer = IntegratedGradients(model)
        artifact_detector = LightweightArtifactDetector(sampling_rate=500)
        print("Resources loaded.")
    except Exception as e:
        print(f"Error loading resources: {e}")

async def send_alert(payload: Dict):
    """Sends an alert to the configured webhook."""
    if not WEBHOOK_URL:
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(WEBHOOK_URL, json=payload, timeout=5.0)
    except Exception as e:
        print(f"Failed to send alert: {e}")

async def log_feedback_to_bq(feedback: FeedbackRequest):
    """Logs feedback to BigQuery asynchronously."""
    # In a real app, use google-cloud-bigquery
    # Here we just log to stdout for the demo, which Cloud Logging picks up
    entry = {
        "event_type": "feedback",
        "feedback_id": f"fb-{datetime.utcnow().timestamp()}",
        "prediction_id": feedback.prediction_id,
        "patient_id": feedback.patient_id,
        "model_version": feedback.model_version,
        "predicted_label": feedback.predicted_label,
        "corrected_label": feedback.corrected_label,
        "cardiologist_id": feedback.cardiologist_id,
        "comments": feedback.comments,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "PENDING"
    }
    audit_logger.info(json.dumps(entry))

@app.on_event("startup")
async def startup_event():
    load_resources()

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """Endpoint for cardiologists to submit corrections."""
    background_tasks.add_task(log_feedback_to_bq, feedback)
    return {"status": "received", "message": "Feedback submitted for validation"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ECGRequest, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prediction_id = f"pred-{datetime.utcnow().timestamp()}"
        
        # Preprocess Input
        signal = np.array(request.signal, dtype=np.float32)
        if len(signal.shape) == 2:
            signal = np.expand_dims(signal, axis=0) # (1, 5000, 12)
            
        # 1. Quality Check
        lead_ii = signal[0, :, 1]
        beat_results = artifact_detector.process_record(lead_ii)
        valid_beats = sum(1 for b in beat_results if b['is_valid'])
        total_beats = len(beat_results)
        quality_score = (valid_beats / total_beats * 100) if total_beats > 0 else 0.0
        
        # 2. Inference (MC Dropout)
        mc_predictions = []
        for _ in range(request.mc_samples):
            preds = model(signal, training=True)
            if isinstance(preds, dict):
                p_preds = preds['pathology'].numpy()
            else:
                p_preds = preds.numpy()
            mc_predictions.append(p_preds)
            
        # Stack: (Samples, Batch, Classes)
        mc_predictions = np.array(mc_predictions)
        
        # If Batch=1, squeeze axis 1. If Batch>1, keep it (but logic below assumes single record)
        if mc_predictions.shape[1] == 1:
            mc_predictions = mc_predictions.squeeze(axis=1)
        else:
            # If batch > 1, we probably shouldn't be here given the request structure, 
            # but let's handle it or fail. 
            # The current logic below (percentile) works on axis 0 (Samples).
            # So mc_predictions would be (Samples, Batch, Classes).
            # mean(axis=0) -> (Batch, Classes).
            # But the response model expects simple Dict[str, float] for diagnosis.
            # This implies the API only supports Batch=1.
            # We'll force squeeze if it's 1, otherwise raise warning.
            pass

        mean_preds = np.mean(mc_predictions, axis=0)
        std_preds = np.std(mc_predictions, axis=0)
        ci_lower = np.percentile(mc_predictions, 2.5, axis=0)
        ci_upper = np.percentile(mc_predictions, 97.5, axis=0)
        
        # Handle Batch > 1 case for response (Just take first item if batch provided, or error)
        if len(mean_preds.shape) > 1:
             # (Batch, Classes) -> Take 0th
             mean_preds = mean_preds[0]
             std_preds = std_preds[0]
             ci_lower = ci_lower[0]
             ci_upper = ci_upper[0]
        
        classes = ["Normal", "AFIB", "MI", "PVC", "Noise"]
        diagnosis_dict = {c: float(mean_preds[i]) for i, c in enumerate(classes)}
        
        # 3. Triage Logic
        triage_status = "ROUTINE"
        triage_reasons = []
        
        # Check MI (Critical)
        if diagnosis_dict["MI"] > TRIAGE_THRESHOLDS["MI"]:
            triage_status = "CRITICAL"
            triage_reasons.append(f"MI Risk {diagnosis_dict['MI']:.2f} > {TRIAGE_THRESHOLDS['MI']}")
            
        # Check AFIB (Urgent) - Only if not already Critical
        if triage_status != "CRITICAL" and diagnosis_dict["AFIB"] > TRIAGE_THRESHOLDS["AFIB"]:
            triage_status = "URGENT"
            triage_reasons.append(f"AFIB Risk {diagnosis_dict['AFIB']:.2f} > {TRIAGE_THRESHOLDS['AFIB']}")
            
        triage_result = TriageResult(
            status=triage_status,
            reason="; ".join(triage_reasons) if triage_reasons else "No urgent findings",
            action_required=(triage_status in ["URGENT", "CRITICAL"])
        )

        # 4. Explanation (Optional)
        explanations = None
        if request.explain:
            top_class_idx = np.argmax(mean_preds)
            saliency = explainer.explain(signal[0], top_class_idx, m_steps=25) 
            lead_importance = np.mean(np.abs(saliency), axis=0).tolist()
            leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            top_lead = leads[np.argmax(lead_importance)]
            
            explanations = {
                "lead_importance": lead_importance,
                "top_contributing_lead": top_lead
            }

        # 5. Audit Logging & Alerting
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_id": prediction_id,
            "patient_id": request.patient_id,
            "triage_status": triage_status,
            "diagnosis": diagnosis_dict,
            "quality_score": quality_score,
            "model_version": "v1-hybrid"
        }
        audit_logger.info(json.dumps(audit_entry))
        
        if triage_result.action_required:
            background_tasks.add_task(send_alert, audit_entry)

        # Response
        return {
            "diagnosis": diagnosis_dict,
            "uncertainty": {c: float(std_preds[i]) for i, c in enumerate(classes)},
            "confidence_intervals": {c: [float(ci_lower[i]), float(ci_upper[i])] for i, c in enumerate(classes)},
            "quality_metrics": {
                "quality_score": float(quality_score),
                "valid_beats": float(valid_beats),
                "total_beats": float(total_beats)
            },
            "triage": triage_result,
            "explanations": explanations,
            "metadata": {
                "model_version": "v1-hybrid",
                "timestamp": datetime.utcnow().isoformat()
            },
            "prediction_id": prediction_id
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
