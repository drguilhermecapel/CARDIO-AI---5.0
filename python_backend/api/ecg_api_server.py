import time
import uuid
import logging
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI, Depends, HTTPException, status, Request, UploadFile, File, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import internal services
# Assuming these exist from previous steps or placeholders
from integration.cardio_ai_service import CardioAIIntegrationService
from infrastructure.audit_logger import SecureAuditLogger

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ECG_API")

# --- Configuration ---
SECRET_KEY = "YOUR_SUPER_SECRET_KEY_CHANGE_IN_PROD"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# --- Security & Auth ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = data.copy()
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- Models ---
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str = "physician"

class PatientData(BaseModel):
    id: str
    age: int = Field(..., gt=0, lt=120)
    sex: str = Field(..., pattern="^(Male|Female|Other)$")
    symptoms: List[str] = []
    history: List[str] = []
    
    # PII fields should be handled carefully (encrypted in transit)
    name_encrypted: Optional[str] = None 

class ECGRequest(BaseModel):
    patient: PatientData
    signal_data: List[List[float]] # 12 leads x N samples
    sampling_rate: int = Field(500, ge=100, le=1000)
    device_id: str
    
    @validator('signal_data')
    def validate_signal(cls, v):
        if len(v) != 12:
            raise ValueError('ECG must have 12 leads')
        if len(v[0]) < 1000:
            raise ValueError('Signal too short (< 2 seconds)')
        return v

class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    status: str
    diagnosis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    clinical_actions: List[str]
    quality_score: float
    
# --- App Setup ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="CardioAI Clinical API",
    description="HIPAA-Compliant ECG Analysis API",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hospital-system.internal"], # Restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["api.cardioai.com", "localhost", "127.0.0.1"]
)

# Services
# In a real app, these would be injected dependencies
integration_service = CardioAIIntegrationService()
audit_logger = SecureAuditLogger(log_dir="logs/api_audit")

# --- Dependencies ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Mock user lookup
    user = User(username=username, role="cardiologist")
    if user is None:
        raise credentials_exception
    return user

async def verify_api_key(request: Request):
    # For machine-to-machine auth (e.g. ECG carts)
    api_key = request.headers.get("X-API-Key")
    if api_key != "VALID_API_KEY_123": # Mock check
        # Allow if user token is present, else fail
        if "Authorization" not in request.headers:
             raise HTTPException(status_code=403, detail="Invalid API Key")

# --- Endpoints ---

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy", 
        "version": "5.0.0", 
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/auth/token", response_model=Token, tags=["Auth"])
@limiter.limit("5/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    # Mock DB check
    # user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if form_data.username != "admin" or form_data.password != "secret":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username, "role": "admin"}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": form_data.username})
    
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/analyze", response_model=AnalysisResponse, tags=["Clinical"])
@limiter.limit("10/minute")
async def analyze_ecg(
    request: Request,
    ecg_request: ECGRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Performs full clinical analysis of 12-lead ECG.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    logger.info(f"Received Analysis Request {request_id} from {current_user.username}")
    
    try:
        # 1. Convert Pydantic to Dict/Numpy
        import numpy as np
        signal = np.array(ecg_request.signal_data)
        meta = ecg_request.patient.dict()
        
        # 2. Process
        result = integration_service.process_ecg(signal, meta)
        
        # 3. Handle Rejection
        if result['status'] == "REJECTED":
            raise HTTPException(status_code=400, detail=f"ECG Rejected: {result.get('error')}")
            
        # 4. Audit Log (Background Task to not block response)
        background_tasks.add_task(
            audit_logger.log_event,
            event_type="API_ANALYSIS",
            user_id=current_user.username,
            action="ANALYZE_ECG",
            resource=request_id,
            details={
                "patient_id": ecg_request.patient.id,
                "diagnosis": result['diagnosis']['diagnosis'],
                "processing_time": time.time() - start_time
            }
        )
        
        # 5. Construct Response
        response = AnalysisResponse(
            analysis_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            status="COMPLETED",
            diagnosis=result['diagnosis'],
            risk_assessment=result['risk_profile'],
            clinical_actions=[result['recommendation']],
            quality_score=result['quality_check']['score']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Analysis Error")

@app.post("/upload/dicom", tags=["Clinical"])
async def upload_dicom(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Uploads a DICOM ECG file for processing.
    """
    if not file.filename.endswith('.dcm'):
        raise HTTPException(status_code=400, detail="Invalid file format. Must be DICOM (.dcm)")
    
    # Logic to parse DICOM and extract signal would go here
    # import pydicom
    # ds = pydicom.dcmread(file.file)
    # ...
    
    return {"filename": file.filename, "status": "Uploaded", "message": "Processing queued"}

# Run with: uvicorn api.ecg_api_server:app --reload
