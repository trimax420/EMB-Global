import os
from pathlib import Path
from typing import List, ClassVar, Optional
from pydantic_settings import BaseSettings

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 3 levels up: src/app/core -> src/app -> src
print(f"Base directory: {BASE_DIR}")

# Define absolute paths
def get_abs_path(rel_path: str) -> Path:
    return BASE_DIR / rel_path

class Settings(BaseSettings):
    """Application settings"""
    # Base settings
    APP_NAME: str = "Security Dashboard API"
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    
    # CORS settings - explicitly allow all for development
    CORS_ORIGINS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    CORS_METHODS: List[str] = ["*"]
    ALLOW_CREDENTIALS: bool = True
    
    # Storage paths
    UPLOAD_DIR: str = "uploads"
    PROCESSED_DIR: str = "processed"
    SCREENSHOTS_DIR: str = "screenshots"
    THUMBNAILS_DIR: str = "thumbnails"
    FRAMES_DIR: str = "frames"
    ALERTS_DIR: str = "alerts"
    
    # Public videos path
    PUBLIC_VIDEOS_DIR: Path = Path("E:/code/EMB Global/public/videos")
    
    # Database
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "security_dashboard")
    DATABASE_URL: Optional[str] = None
    
    # Model directories
    MODEL_DIR: str = "models"
    
    # AI Model paths
    FACE_MODEL_PATH: Path = BASE_DIR / "models" / "yolov8n-face.pt"
    POSE_MODEL_PATH: Path = BASE_DIR / "models" / "yolov8n-pose.pt"
    OBJECT_MODEL_PATH: Path = BASE_DIR / "models" / "yolov5s.pt"
    FACE_EXTRACTION_MODEL_PATH: Path = BASE_DIR / "models" / "yolov8n-face.pt"
    LOITERING_MODEL_PATH: Path = BASE_DIR / "models" / "yolov5s.pt"
    THEFT_MODEL_PATH: Path = BASE_DIR / "models" / "yolo11n-pose.pt"
    
    # Inference settings
    DEFAULT_CONFIDENCE: float = 0.5
    
    # Video processing settings
    MAX_WORKERS: int = 2
    TARGET_PROCESSING_FPS: int = 15
    DETECTION_THRESHOLD: float = 0.4
    FRAME_BUFFER_SIZE: int = 32
    UPDATE_INTERVAL: float = 0.1
    STATS_UPDATE_INTERVAL: float = 2.0
    
    # Theft detection settings
    SKIP_FRAMES: int = 5
    HAND_STAY_TIME_CHEST: float = 2.0  # seconds before suspicion (chest region)
    HAND_STAY_TIME_WAIST: float = 3.0  # seconds for waist region
    CROP_PADDING: int = 50  # Padding for crop area to include more details
    OBJECT_PROXIMITY_THRESHOLD: int = 50  # Max distance to consider object proximity
    
    # WebSocket settings
    WEBSOCKET_ENABLED: bool = True
    
    # Loitering detection settings
    LOITERING_THRESHOLD: float = 10.0  # Time threshold for loitering in seconds
    
    # Set DATABASE_URL based on other DB settings if not explicitly provided
    def __init__(self, **data):
        super().__init__(**data)
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings
settings = Settings()

# Remove the duplicate global variables since they're now in the Settings class