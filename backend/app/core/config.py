from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    THUMBNAILS_DIR: Path = DATA_DIR / "thumbnails"
    SCREENSHOTS_DIR: Path = DATA_DIR / "screenshots"
    INCIDENTS_DIR: Path = DATA_DIR / "incidents"
    
    # Video files directory
    ROOT_DIR: Path = BASE_DIR
    VIDEO_FILES_DIR: str = os.getenv("VIDEO_FILES_DIR", str(BASE_DIR / "videos"))
    MOCK_FRAME_WIDTH: int = 640
    MOCK_FRAME_HEIGHT: int = 480

    # Model directories and paths
    MODELS_DIR: Path = BASE_DIR / "models"
    FACE_MODEL_PATH: Path = MODELS_DIR / "face_detection.pt"
    POSE_MODEL_PATH: Path = MODELS_DIR / "pose_detection.pt"
    OBJECT_MODEL_PATH: Path = MODELS_DIR / "object_detection.pt"

    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Video processing settings
    SKIP_FRAMES: int = 2  # Process every Nth frame
    VIDEO_QUALITY: int = 85  # JPEG compression quality for frames
    TARGET_FPS: int = 15  # Target FPS for realtime streaming

    # Detection settings
    DETECTION_THRESHOLD: float = 0.6  # Default confidence threshold

    # WebSocket settings
    PING_INTERVAL: int = 30  # Seconds between WebSocket ping messages
    STALE_CONNECTION_TIMEOUT: int = 300  # Seconds after which a connection is considered stale
    MAX_FRAME_WIDTH: int = 640  # Maximum width for streamed frames

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()