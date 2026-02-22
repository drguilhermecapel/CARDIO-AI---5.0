import pytest
import os
import json
from infrastructure.audit_logger import SecureAuditLogger

@pytest.fixture
def secure_logger(tmpdir):
    log_dir = str(tmpdir.mkdir("audit_logs"))
    return SecureAuditLogger(log_dir=log_dir)

def test_log_creation(secure_logger):
    secure_logger.log_event(
        "TEST_EVENT", "user1", "TEST_ACTION", "RES_1", 
        {"data": "test"}
    )
    
    assert os.path.exists(secure_logger.log_file)
    
    with open(secure_logger.log_file, "r") as f:
        line = f.readline()
        entry = json.loads(line)
        assert "encrypted_data" in entry
        assert "signature" in entry
        assert "hash" in entry

def test_anonymization(secure_logger):
    details = {"patient_name": "John Doe", "diagnosis": "Flu"}
    
    # We need to decrypt to check anonymization, so let's peek into the implementation
    # or rely on verify_log_integrity to decrypt implicitly?
    # Let's decrypt manually for test
    
    secure_logger.log_event("TEST", "u1", "a1", "r1", details)
    
    with open(secure_logger.log_file, "r") as f:
        line = f.readline()
        entry = json.loads(line)
        
        # Decrypt
        import base64
        enc = base64.b64decode(entry['encrypted_data'])
        dec = secure_logger.cipher.decrypt(enc)
        payload = json.loads(dec)
        
        assert payload['details']['patient_name'] != "John Doe"
        assert len(payload['details']['patient_name']) == 64 # SHA256 hex
        assert payload['details']['diagnosis'] == "Flu"

def test_integrity_verification(secure_logger):
    # Log 3 events
    secure_logger.log_event("E1", "u1", "a1", "r1", {})
    secure_logger.log_event("E2", "u1", "a1", "r1", {})
    secure_logger.log_event("E3", "u1", "a1", "r1", {})
    
    assert secure_logger.verify_log_integrity(secure_logger.log_file) is True
    
    # Tamper with file
    with open(secure_logger.log_file, "r") as f:
        lines = f.readlines()
        
    # Modify middle entry
    entry = json.loads(lines[1])
    entry['hash'] = "0" * 64 # Break hash
    lines[1] = json.dumps(entry) + "\n"
    
    with open(secure_logger.log_file, "w") as f:
        f.writelines(lines)
        
    assert secure_logger.verify_log_integrity(secure_logger.log_file) is False
