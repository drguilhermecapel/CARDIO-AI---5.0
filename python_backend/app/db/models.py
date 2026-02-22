from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
import datetime
import uuid

Base = declarative_base()

def get_db():
    # Mock DB session for dependency injection
    # In prod: yield SessionLocal()
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String) # cardiologist, nurse, admin
    is_active = Column(Boolean, default=True)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    dob = Column(String)
    sex = Column(String)
    mrn = Column(String, unique=True, index=True) # Medical Record Number
    
    ecg_records = relationship("ECGRecord", back_populates="patient")

class ECGRecord(Base):
    __tablename__ = "ecg_records"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"))
    device_id = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Raw Data
    raw_data_path = Column(String) # S3 path
    sample_rate = Column(Integer)
    
    # Analysis Results
    status = Column(String) # PROCESSING, COMPLETED, FAILED
    diagnosis_main = Column(String)
    confidence = Column(Float)
    report_json = Column(JSON) # Full structured report
    
    # Review
    is_reviewed = Column(Boolean, default=False)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_notes = Column(Text, nullable=True)
    
    # Relationships
    patient = relationship("Patient", back_populates="ecg_records")
    audit_logs = relationship("AuditLog", back_populates="ecg_record")

class AuditLog(Base):
    """
    Immutable Audit Trail for Regulatory Compliance (FDA/HIPAA).
    """
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    ecg_record_id = Column(String, ForeignKey("ecg_records.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    event_type = Column(String) # ANALYSIS_GENERATED, PHYSICIAN_REVIEW, SYSTEM_ERROR
    model_version = Column(String)
    
    # Data Snapshots
    input_snapshot_hash = Column(String) # SHA256 of input signal
    output_snapshot = Column(JSON) # The diagnosis/recommendation generated
    
    # Verification
    verified_outcome = Column(JSON, nullable=True) # Ground truth added later
    
    # Metadata
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    ip_address = Column(String, nullable=True)
    
    ecg_record = relationship("ECGRecord", back_populates="audit_logs")

# Database Setup (SQLite for demo)
SQLALCHEMY_DATABASE_URL = "sqlite:///./cardioai.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
