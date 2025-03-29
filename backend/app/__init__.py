from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Security Dashboard API",
        description="API for security monitoring, alerts and analytics",
        version="1.0.0"
    )

    # CORS settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api")

    return app 