import uvicorn
import logging
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.config import settings
from database import init_db
from app.directory_initializer import ensure_app_directories

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    app = FastAPI(
        title="Security Dashboard API",
        description="API for security monitoring, alerts and analytics",
        version="1.0.0"
    )

    # CORS settings
    origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router, prefix="/api")

    return app

app = create_app()

@app.on_event("startup")
async def startup_event():
    """Initialize application resources during startup"""
    try:
        # Step 1: Ensure directories exist
        logger.info("Initializing application directories...")
        ensure_app_directories(settings)
        logger.info("Application directories initialized successfully")
        
        # Step 2: Initialize database
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        # Detailed error logging
        error_msg = f"Critical error during startup: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        # Re-raise to prevent app from starting with incomplete initialization
        raise

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        # Increase timeout for startup
        timeout_keep_alive=120,
        log_level="info"
    )