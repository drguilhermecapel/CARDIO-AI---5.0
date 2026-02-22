import hashlib
import time
import json
from typing import Dict, Any

class AuditLog:
    def __init__(self, log_file="audit_log.jsonl"):
        self.log_file = log_file

    def log_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """
        Log an event immutably.
        """
        timestamp = time.time()
        # Create a hash chain for immutability verification
        prev_hash = self._get_last_hash()
        
        entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
            "prev_hash": prev_hash
        }
        
        # Calculate hash of current entry
        entry_string = json.dumps(entry, sort_keys=True)
        current_hash = hashlib.sha256(entry_string.encode()).hexdigest()
        entry["hash"] = current_hash
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
        return current_hash

    def _get_last_hash(self):
        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                if not lines:
                    return "0" * 64 # Genesis hash
                last_line = json.loads(lines[-1])
                return last_line.get("hash", "0" * 64)
        except FileNotFoundError:
            return "0" * 64

class FeedbackSystem:
    def __init__(self, audit_log: AuditLog):
        self.audit_log = audit_log

    def submit_feedback(self, case_id: str, user_id: str, correction: Dict[str, Any]):
        """
        Submit clinician feedback for a specific case.
        """
        # 1. Log the feedback
        self.audit_log.log_event(
            event_type="CLINICIAN_FEEDBACK",
            user_id=user_id,
            details={
                "case_id": case_id,
                "correction": correction,
                "status": "PENDING_REVIEW"
            }
        )
        
        # 2. Trigger retraining pipeline if threshold met (Mock)
        # check_retraining_trigger()
        
        return {"status": "success", "message": "Feedback received and logged."}

# Example Usage
if __name__ == "__main__":
    audit = AuditLog()
    feedback = FeedbackSystem(audit)
    
    # Simulate a diagnosis
    audit.log_event("DIAGNOSIS_GENERATED", "system", {"case_id": "123", "diagnosis": "STEMI"})
    
    # Simulate clinician correction
    feedback.submit_feedback("123", "dr_smith", {"diagnosis": "Pericarditis", "reason": "Diffuse STE, PR depression"})
    
    print("Audit log updated.")
