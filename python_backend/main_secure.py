from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

from security.auth import AuthManager, Token, User, fake_users_db
from security.encryption import EncryptionManager
from security.audit import audit_logger
from security.rbac import RoleChecker

app = FastAPI(title="CardioAI Secure API", version="5.0.0")

# Initialize Crypto
# In prod, load key from secure env var
crypto = EncryptionManager() 

# --- Models ---
class MFAVerification(BaseModel):
    username: str
    code: str

class PatientData(BaseModel):
    name: str
    ecg_data: str # Base64
    notes: str

# --- Auth Routes ---

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), request: Request = None):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        # Log failed attempt
        audit_logger.log_action("unknown", "LOGIN_FAILED", "auth", request.client.host, "FAILURE")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    user = User(**user_dict)
    if not AuthManager.verify_password(form_data.password, user.hashed_password):
        audit_logger.log_action(user.username, "LOGIN_FAILED", "auth", request.client.host, "FAILURE")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    # Issue Partial Token (MFA Pending)
    access_token = AuthManager.create_access_token(
        data={"sub": user.username, "role": user.role, "mfa_verified": False}
    )
    
    audit_logger.log_action(user.username, "LOGIN_PARTIAL", "auth", request.client.host, "SUCCESS")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/verify-mfa", response_model=Token)
async def verify_mfa_code(verification: MFAVerification, request: Request):
    user_dict = fake_users_db.get(verification.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="User not found")
    
    user = User(**user_dict)
    
    if AuthManager.verify_mfa(user.mfa_secret, verification.code):
        # Issue Full Token
        access_token = AuthManager.create_access_token(
            data={"sub": user.username, "role": user.role, "mfa_verified": True}
        )
        audit_logger.log_action(user.username, "MFA_SUCCESS", "auth", request.client.host, "SUCCESS")
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        audit_logger.log_action(user.username, "MFA_FAILED", "auth", request.client.host, "FAILURE")
        raise HTTPException(status_code=400, detail="Invalid MFA Code")

# --- Protected Routes ---

@app.post("/patients/record", dependencies=[Depends(RoleChecker(["cardiologist", "admin"]))])
async def create_patient_record(data: PatientData, request: Request, current_user: User = Depends(AuthManager.get_current_user)):
    # 1. Encrypt Sensitive Data
    encrypted_name = crypto.encrypt(data.name)
    encrypted_notes = crypto.encrypt(data.notes)
    
    # 2. Save to DB (Mock)
    record_id = "rec_12345"
    # db.save(...)
    
    # 3. Audit Log
    audit_logger.log_action(
        current_user[0].username, 
        "CREATE_RECORD", 
        f"patient_record:{record_id}", 
        request.client.host, 
        "SUCCESS"
    )
    
    return {"status": "created", "id": record_id, "encrypted_name_preview": encrypted_name[:20] + "..."}

@app.get("/admin/audit-logs", dependencies=[Depends(RoleChecker(["admin"]))])
async def view_audit_logs(request: Request, current_user: User = Depends(AuthManager.get_current_user)):
    # In reality, read from the secure log file/DB
    audit_logger.log_action(current_user[0].username, "VIEW_LOGS", "audit_system", request.client.host, "SUCCESS")
    return {"message": "Audit logs retrieved", "logs": "..."}

@app.get("/health")
def health():
    return {"status": "secure"}
