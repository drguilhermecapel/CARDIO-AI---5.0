from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pydantic import BaseModel
import pyotp

# Configuration
SECRET_KEY = "YOUR_SECURE_SECRET_KEY_IN_ENV"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock User DB
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@hospital.com",
        "hashed_password": pwd_context.hash("secret"),
        "role": "admin",
        "disabled": False,
        "mfa_secret": pyotp.random_base32() # Generate for demo
    },
    "cardiologist": {
        "username": "cardiologist",
        "full_name": "Dr. Heart",
        "email": "dr.heart@hospital.com",
        "hashed_password": pwd_context.hash("heart123"),
        "role": "cardiologist",
        "disabled": False,
        "mfa_secret": pyotp.random_base32()
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    mfa_verified: bool = False

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str
    mfa_secret: Optional[str] = None # In real DB, keep this separate/encrypted

class UserInDB(User):
    hashed_password: str

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
    async def get_current_user(token: str = Depends(oauth2_scheme)):
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
        except JWTError:
            raise credentials_exception
            
        user_dict = fake_users_db.get(username)
        if user_dict is None:
            raise credentials_exception
        user = UserInDB(**user_dict)
        
        if user.disabled:
            raise HTTPException(status_code=400, detail="Inactive user")
            
        return user, token_data

    @staticmethod
    def verify_mfa(secret: str, code: str) -> bool:
        totp = pyotp.TOTP(secret)
        return totp.verify(code)
