import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base, AuditLog, ECGRecord
from audit.audit_logger import AuditLogger

# Setup In-Memory DB for Tests
engine = create_engine("sqlite:///:memory:")
SessionLocal = sessionmaker(bind=engine)

@pytest.fixture(scope="module")
def db():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def logger(db):
    return AuditLogger(db)

def test_log_analysis(logger, db):
    # Create Dummy Record
    rec = ECGRecord(id="REC_001", patient_id="P_001")
    db.add(rec)
    db.commit()
    
    input_data = [0.1, 0.2, 0.3]
    result = {"diagnosis": "Normal", "confidence": 0.99}
    
    logger.log_analysis_event("REC_001", "v1.0", input_data, result)
    
    log = db.query(AuditLog).filter(AuditLog.ecg_record_id == "REC_001").first()
    assert log is not None
    assert log.event_type == "ANALYSIS_GENERATED"
    assert log.model_version == "v1.0"
    assert log.output_snapshot['diagnosis'] == "Normal"

def test_log_review(logger, db):
    logger.log_review_event("REC_001", 101, "Normal", "AFib", "Missed P-waves")
    
    log = db.query(AuditLog).filter(AuditLog.event_type == "PHYSICIAN_REVIEW").first()
    assert log is not None
    assert log.output_snapshot['concordance'] is False
    assert log.output_snapshot['final_diagnosis'] == "AFib"

def test_export_logs(logger, db):
    start = datetime.utcnow() - timedelta(hours=1)
    end = datetime.utcnow() + timedelta(hours=1)
    
    json_logs = logger.export_logs(start, end)
    assert "REC_001" in json_logs
    assert "ANALYSIS_GENERATED" in json_logs
