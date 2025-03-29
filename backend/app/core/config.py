from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "processed"
    FRAMES_DIR: Path = BASE_DIR / "frames"
    ALERTS_DIR: Path = BASE_DIR / "alerts"
    THUMBNAILS_DIR: Path = BASE_DIR / "thumbnails"
    MODELS_DIR: Path = BASE_DIR / "models"
    SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"  # New directory for theft screenshots

    # Model paths
    FACE_MODEL_PATH: Path = BASE_DIR / "models/yolov8n-face.pt"
    POSE_MODEL_PATH: Path = BASE_DIR / "models/yolov8n-pose.pt"
    OBJECT_MODEL_PATH: Path = BASE_DIR / "models/yolov5s.pt"
    FACE_EXTRACTION_MODEL_PATH: Path = BASE_DIR / "models/yolov8n-face.pt"
    LOITERING_MODEL_PATH: Path = BASE_DIR / "models/yolov5s.pt"
    THEFT_MODEL_PATH: Path = BASE_DIR / "models/yolo11n-pose.pt"

    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Video processing settings
    MAX_WORKERS: int = 2
    TARGET_PROCESSING_FPS: int = 15
    DETECTION_THRESHOLD: float = 0.4
    FRAME_BUFFER_SIZE: int = 32
    UPDATE_INTERVAL: float = 0.1
    STATS_UPDATE_INTERVAL: float = 2.0
    
    # Theft detection settings
    SKIP_FRAMES: int = 3
    HAND_STAY_TIME_CHEST: float = 1.0  # seconds before suspicion (chest region)
    HAND_STAY_TIME_WAIST: float = 1.5  # seconds for waist region
    CROP_PADDING: int = 50  # Padding for crop area to include more details
    OBJECT_PROXIMITY_THRESHOLD: int = 50  # Max distance to consider object proximity

    class Config:
        env_file = ".env"

settings = Settings()