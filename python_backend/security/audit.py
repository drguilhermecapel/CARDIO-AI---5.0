import logging
from infrastructure.audit_logger import SecureAuditLogger

# Wrapper to expose a simple interface compatible with main_secure.py
class AuditWrapper:
    def __init__(self):
        self.logger = SecureAuditLogger(log_dir="logs/secure_audit")

    def log_action(self, user: str, action: str, resource: str, ip: str, outcome: str):
        self.logger.log_event(
            event_type="SECURITY_ACTION",
            user_id=user,
            action=action,
            resource=resource,
            details={"ip_address": ip},
            outcome=outcome
        )

audit_logger = AuditWrapper()
