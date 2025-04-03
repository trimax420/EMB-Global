import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.config import settings
from database import init_db
from app.directory_initializer import ensure_app_directories
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    app = FastAPI(
        title="Security Dashboard API",
        description="API for security video analytics and detection",
        version="1.0.0"
    )

    # CORS settings - explicitly allow WebSocket
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve the static demo files for websocket testing
    static_dir = Path(__file__).parent / "app" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include API router only once with the /api prefix
    app.include_router(api_router, prefix="/api")

    # Create root redirector
    @app.get("/")
    async def redirect_to_api():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/api")

    return app

app = create_app()

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Ensuring application directories exist...")
        ensure_app_directories(settings)  # Pass the settings object as an argument
        
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialized successfully")
        
        # Print available endpoints for debugging
        for route in app.routes:
            logger.info(f"Route: {route.path}, Methods: {getattr(route, 'methods', ['WebSocket'])}")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Print the full traceback
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )