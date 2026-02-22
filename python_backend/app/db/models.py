from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String) # cardiologist, admin, system
    is_active = Column(Boolean, default=True)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(String, primary_key=True, index=True) # MRN
    name_encrypted = Column(String)
    dob = Column(DateTime)
    sex = Column(String)

class ECGRecord(Base):
    __tablename__ = "ecg_records"
    id = Column(String, primary_key=True, index=True) # UUID
    patient_id = Column(String, ForeignKey("patients.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Metadata
    device_id = Column(String)
    sample_rate = Column(Integer)
    leads = Column(Integer, default=12)
    
    # Data Storage (Path to S3/Blob or Raw JSON for demo)
    raw_data_path = Column(String) 
    
    # Analysis Results
    status = Column(String, default="PENDING") # PENDING, PROCESSING, COMPLETED, FAILED
    diagnosis_main = Column(String)
    confidence = Column(Float)
    report_json = Column(JSON) # Full structured report
    
    # Review
    is_reviewed = Column(Boolean, default=False)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    review_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    patient = relationship("Patient")
    reviewer = relationship("User")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
