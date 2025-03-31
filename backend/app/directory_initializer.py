
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def ensure_app_directories(settings):
    """
    Ensures all required application directories exist.
    Creates them if they don't exist.
    
    Args:
        settings: Application settings containing directory paths
    
    Returns:
        bool: True if all directories were verified/created successfully
    """
    try:
        # List of directories to ensure exist
        directories = [
            settings.UPLOAD_DIR,
            settings.PROCESSED_DIR,
            settings.FRAMES_DIR,
            settings.ALERTS_DIR,
            settings.THUMBNAILS_DIR,
            settings.MODELS_DIR,
            settings.SCREENSHOTS_DIR
        ]
        
        # Create directories if they don't exist
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            else:
                logger.info(f"Directory already exists: {directory}")
        
        # Create subdirectories for specific purposes
        videos_upload_dir = settings.UPLOAD_DIR / "videos"
        videos_upload_dir.mkdir(parents=True, exist_ok=True)
        
        raw_videos_dir = settings.UPLOAD_DIR / "raw"
        raw_videos_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating application directories: {str(e)}")
        raise