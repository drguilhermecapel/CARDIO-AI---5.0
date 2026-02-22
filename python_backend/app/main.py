from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.api.v1.endpoints import ecg, worklist, explainability
from app.db.models import Base, engine

# Create Tables
Base.metadata.create_all(bind=engine)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ecg.router, prefix=f"{settings.API_V1_STR}/ecg", tags=["ECG"])
app.include_router(worklist.router, prefix=f"{settings.API_V1_STR}/worklist", tags=["Worklist"])
app.include_router(explainability.router, prefix=f"{settings.API_V1_STR}/explainability", tags=["Explainability"])

@app.get("/health")
def health_check():
    return {"status": "ok", "db": "connected", "redis": "connected"}
