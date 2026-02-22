import json
import logging
import os
import datetime
import hashlib
import base64
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger("SecureAudit")

class SecureAuditLogger:
    """
    Secure Audit Logger for ISO 14971 and GDPR/LGPD compliance.
    Features:
    - Structured JSON logging
    - Field-level anonymization
    - AES-256 Encryption (Fernet)
    - RSA Digital Signatures for Non-repudiation
    - Integrity Chaining (Blockchain-like linking)
    """
    
    def __init__(self, 
                 log_dir: str = "logs/audit", 
                 encryption_key: Optional[bytes] = None,
                 private_key_path: Optional[str] = None):
        
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.log_file = os.path.join(log_dir, f"audit_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl")
        
        # Setup Encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            # Generate ephemeral key if none provided (for testing/demo)
            # In prod, load from KMS
            self.cipher = Fernet(Fernet.generate_key())
            
        # Setup Signing
        self.private_key = self._load_or_generate_private_key(private_key_path)
        
        # Chain State
        self.last_hash = "0" * 64

    def _load_or_generate_private_key(self, path: Optional[str]):
        if path and os.path.exists(path):
            with open(path, "rb") as key_file:
                return serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend()
                )
        else:
            # Generate new (for demo purposes)
            return rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )

    def _anonymize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymizes PII fields for GDPR/LGPD.
        Hashes MRN, Patient Name, DOB.
        """
        pii_fields = ['patient_name', 'mrn', 'dob', 'patient_id', 'social_security']
        anonymized = data.copy()
        
        for field in pii_fields:
            if field in anonymized:
                # Salted hash would be better, using simple hash for demo
                val = str(anonymized[field]).encode()
                anonymized[field] = hashlib.sha256(val).hexdigest()
                
        return anonymized

    def _sign_data(self, data_bytes: bytes) -> str:
        """
        Generates RSA digital signature.
        """
        signature = self.private_key.sign(
            data_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def log_event(self, 
                  event_type: str, 
                  user_id: str, 
                  action: str, 
                  resource: str,
                  details: Dict[str, Any],
                  outcome: str = "SUCCESS") -> str:
        """
        Logs a secure audit event.
        """
        timestamp = datetime.datetime.utcnow().isoformat()
        
        # 1. Prepare Payload
        payload = {
            "version": "1.0",
            "event_id": hashlib.uuid.uuid4().hex,
            "timestamp": timestamp,
            "event_type": event_type,
            "actor": {
                "user_id": user_id,
                "role": "system" if user_id == "system" else "user"
            },
            "action": action,
            "resource": resource,
            "outcome": outcome,
            "details": self._anonymize(details),
            "previous_hash": self.last_hash
        }
        
        # 2. Serialize
        payload_json = json.dumps(payload, sort_keys=True)
        payload_bytes = payload_json.encode('utf-8')
        
        # 3. Encrypt (Confidentiality)
        encrypted_payload = self.cipher.encrypt(payload_bytes)
        
        # 4. Sign (Non-repudiation & Integrity)
        signature = self._sign_data(payload_bytes)
        
        # 5. Calculate Hash for Chain
        current_hash = hashlib.sha256(payload_bytes).hexdigest()
        self.last_hash = current_hash
        
        # 6. Write to Log
        log_entry = {
            "timestamp": timestamp,
            "encrypted_data": base64.b64encode(encrypted_payload).decode('utf-8'),
            "signature": signature,
            "hash": current_hash
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.critical(f"AUDIT FAILURE: Could not write to log file. {e}")
            raise
            
        return current_hash

    def verify_log_integrity(self, log_path: str) -> bool:
        """
        Verifies the blockchain-like integrity of the log file.
        """
        if not os.path.exists(log_path):
            return False
            
        prev_hash = "0" * 64
        public_key = self.private_key.public_key()
        
        try:
            with open(log_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    
                    # 1. Decrypt
                    enc_data = base64.b64decode(entry['encrypted_data'])
                    dec_data = self.cipher.decrypt(enc_data)
                    payload = json.loads(dec_data)
                    
                    # 2. Verify Chain
                    if payload['previous_hash'] != prev_hash:
                        logger.error(f"Chain broken at {payload['timestamp']}")
                        return False
                        
                    # 3. Verify Signature
                    sig = base64.b64decode(entry['signature'])
                    try:
                        public_key.verify(
                            sig,
                            dec_data,
                            padding.PSS(
                                mgf=padding.MGF1(hashes.SHA256()),
                                salt_length=padding.PSS.MAX_LENGTH
                            ),
                            hashes.SHA256()
                        )
                    except Exception:
                        logger.error(f"Invalid signature at {payload['timestamp']}")
                        return False
                        
                    # Update hash
                    prev_hash = entry['hash']
                    
            return True
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

# Example Usage
if __name__ == "__main__":
    # Initialize
    audit = SecureAuditLogger()
    
    # Log Diagnostic Event
    audit.log_event(
        event_type="DIAGNOSTIC_ANALYSIS",
        user_id="dr_house",
        action="EXECUTE_MODEL",
        resource="ECG_12345",
        details={
            "patient_id": "P_999",
            "model_version": "v5.0.1",
            "diagnosis": "STEMI",
            "confidence": 0.98,
            "processing_time_ms": 124
        }
    )
    
    # Log Review
    audit.log_event(
        event_type="CLINICAL_REVIEW",
        user_id="dr_house",
        action="CONFIRM_DIAGNOSIS",
        resource="ECG_12345",
        details={
            "notes": "Agreed with AI.",
            "patient_name": "John Doe" # Will be anonymized
        }
    )
    
    # Verify
    is_valid = audit.verify_log_integrity(audit.log_file)
    print(f"Log Integrity Valid: {is_valid}")
