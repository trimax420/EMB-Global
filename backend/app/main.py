from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.api.endpoints import detection, analytics, alerts, auth
from app.core.database import get_db
from app.core.config import settings

app = FastAPI(
    title="Security Surveillance API",
    description="API for real-time security surveillance and analytics",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(detection.router, prefix="/api/detection", tags=["detection"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])

@app.get("/")
def read_root():
    return {"message": "Security Surveillance API"}

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    # Basic health check to ensure DB connection is working
    return {"status": "healthy", "database": "connected"}
