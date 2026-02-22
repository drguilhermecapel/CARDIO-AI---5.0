import hashlib
import json
import time
import logging
from typing import Dict, Any
import hmac

# Configure standard logging
logging.basicConfig(
    filename="audit_trail.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AuditLogger:
    """
    Implements an immutable, tamper-evident audit log.
    """
    def __init__(self, secret_key: str = "audit-signing-key"):
        self.secret_key = secret_key.encode()
        self.last_hash = "0" * 64 # Genesis hash

    def _sign_entry(self, entry_str: str) -> str:
        """Generates HMAC-SHA256 signature for the entry."""
        return hmac.new(self.secret_key, entry_str.encode(), hashlib.sha256).hexdigest()

    def log_action(self, user_id: str, action: str, resource: str, ip_address: str, status: str, details: Dict[str, Any] = None):
        """
        Logs a user action with a cryptographic link to the previous entry.
        """
        timestamp = time.time()
        
        entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "ip_address": ip_address,
            "status": status,
            "details": details or {},
            "prev_hash": self.last_hash
        }
        
        # Canonical JSON string for consistent hashing
        entry_str = json.dumps(entry, sort_keys=True)
        
        # Calculate Hash of current entry (Chain)
        current_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["hash"] = current_hash
        
        # Sign the hash (Integrity)
        entry["signature"] = self._sign_entry(current_hash)
        
        # Update state
        self.last_hash = current_hash
        
        # Write to secure log (In prod, send to SIEM or Write-Once storage)
        logging.info(json.dumps(entry))
        
        return entry

# Singleton
audit_logger = AuditLogger()
