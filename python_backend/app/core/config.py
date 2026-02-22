import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "CardioAI Nexus"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-key-change-in-prod")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "db")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "cardioai")
    DATABASE_URL: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}/{POSTGRES_DB}"
    
    # Redis / Celery
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    CELERY_BROKER_URL: str = f"redis://{REDIS_HOST}:6379/0"
    CELERY_RESULT_BACKEND: str = f"redis://{REDIS_HOST}:6379/0"

    class Config:
        case_sensitive = True

settings = Settings()
