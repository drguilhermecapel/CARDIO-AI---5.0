import os
import jwt
import pyotp
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    mfa_verified: bool = False

class User(BaseModel):
    username: str
    email: str
    full_name: str
    role: str # 'cardiologist', 'reviewer', 'admin'
    hashed_password: str
    mfa_secret: str
    is_active: bool = True

# Mock Database
fake_users_db = {
    "dr_house": {
        "username": "dr_house",
        "email": "house@hospital.com",
        "full_name": "Gregory House",
        "role": "cardiologist",
        "hashed_password": pwd_context.hash("vicodin"),
        "mfa_secret": pyotp.random_base32(),
        "is_active": True,
    },
    "admin_lisa": {
        "username": "admin_lisa",
        "email": "cuddy@hospital.com",
        "full_name": "Lisa Cuddy",
        "role": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "mfa_secret": pyotp.random_base32(),
        "is_active": True,
    }
}

class AuthManager:
    @staticmethod
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password):
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_mfa(secret: str, code: str) -> bool:
        totp = pyotp.TOTP(secret)
        return totp.verify(code)

    @staticmethod
    async def get_current_user(token: str = Security(oauth2_scheme)):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role")
            mfa_verified: bool = payload.get("mfa_verified", False)
            
            if username is None:
                raise credentials_exception
            token_data = TokenData(username=username, role=role, mfa_verified=mfa_verified)
        except jwt.PyJWTError:
            raise credentials_exception
            
        user_dict = fake_users_db.get(username)
        if user_dict is None:
            raise credentials_exception
        return User(**user_dict), token_data

    @staticmethod
    def check_permissions(token_data: TokenData, required_roles: List[str]):
        if not token_data.mfa_verified:
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="MFA verification required"
            )
        if token_data.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
