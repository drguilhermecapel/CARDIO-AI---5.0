import logging
import json
import hashlib
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional

from app.db.models import AuditLog, ECGRecord

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuditLogger")

class AuditLogger:
    """
    Centralized Audit Logging System.
    Ensures all critical events are recorded with full context.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session

    def _hash_data(self, data: Any) -> str:
        """Generates SHA256 hash for data integrity verification."""
        if isinstance(data, (dict, list)):
            s = json.dumps(data, sort_keys=True)
        else:
            s = str(data)
        return hashlib.sha256(s.encode()).hexdigest()

    def log_analysis_event(self, 
                           record_id: str, 
                           model_version: str, 
                           input_data: Any, 
                           result: Dict[str, Any],
                           user_id: Optional[int] = None):
        """
        Logs an AI analysis event.
        """
        try:
            # Create Log Entry
            log_entry = AuditLog(
                ecg_record_id=record_id,
                event_type="ANALYSIS_GENERATED",
                model_version=model_version,
                input_snapshot_hash=self._hash_data(input_data),
                output_snapshot=result,
                user_id=user_id,
                timestamp=datetime.utcnow()
            )
            
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"Audit Log Created: Analysis for Record {record_id}")
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            self.db.rollback()

    def log_review_event(self, 
                         record_id: str, 
                         reviewer_id: int, 
                         original_diagnosis: str,
                         final_diagnosis: str,
                         notes: str):
        """
        Logs a physician review event (agreement or correction).
        """
        try:
            outcome = {
                "original_diagnosis": original_diagnosis,
                "final_diagnosis": final_diagnosis,
                "concordance": original_diagnosis == final_diagnosis,
                "notes": notes
            }
            
            log_entry = AuditLog(
                ecg_record_id=record_id,
                event_type="PHYSICIAN_REVIEW",
                model_version="N/A", # Human review
                input_snapshot_hash="N/A",
                output_snapshot=outcome,
                user_id=reviewer_id,
                timestamp=datetime.utcnow()
            )
            
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"Audit Log Created: Review for Record {record_id}")
            
        except Exception as e:
            logger.error(f"Failed to write review log: {e}")
            self.db.rollback()

    def log_outcome_verification(self, record_id: str, actual_outcome: Dict[str, Any]):
        """
        Logs the ground truth outcome (e.g., from discharge summary or cath lab).
        Essential for continuous learning.
        """
        try:
            # Find original analysis log
            # In a real system, we might link this differently, but here we add a new log
            # or update the verification field of the original log.
            # Adding a new log is safer for immutability.
            
            log_entry = AuditLog(
                ecg_record_id=record_id,
                event_type="OUTCOME_VERIFICATION",
                model_version="N/A",
                input_snapshot_hash="N/A",
                output_snapshot=None,
                verified_outcome=actual_outcome,
                timestamp=datetime.utcnow()
            )
            
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"Audit Log Created: Outcome for Record {record_id}")
            
        except Exception as e:
            logger.error(f"Failed to write outcome log: {e}")
            self.db.rollback()

    def export_logs(self, start_date: datetime, end_date: datetime) -> str:
        """
        Exports logs to JSON for external auditing.
        """
        logs = self.db.query(AuditLog).filter(
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).all()
        
        export_data = []
        for log in logs:
            export_data.append({
                "id": log.id,
                "record_id": log.ecg_record_id,
                "timestamp": log.timestamp.isoformat(),
                "type": log.event_type,
                "model": log.model_version,
                "output": log.output_snapshot,
                "verified": log.verified_outcome
            })
            
        return json.dumps(export_data, indent=2)
