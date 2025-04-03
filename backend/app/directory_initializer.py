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
        # Base directory
        base_dir = Path(__file__).resolve().parent.parent.parent
        
        # Convert string paths to Path objects
        upload_dir = base_dir / settings.UPLOAD_DIR
        processed_dir = base_dir / settings.PROCESSED_DIR
        frames_dir = base_dir / settings.FRAMES_DIR
        alerts_dir = base_dir / settings.ALERTS_DIR
        thumbnails_dir = base_dir / settings.THUMBNAILS_DIR
        screenshots_dir = base_dir / settings.SCREENSHOTS_DIR
        models_dir = base_dir / settings.MODEL_DIR
        
        # List of directories to ensure exist
        directories = [
            upload_dir,
            processed_dir,
            frames_dir,
            alerts_dir,
            thumbnails_dir,
            models_dir,
            screenshots_dir
        ]
        
        # Create directories if they don't exist
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            else:
                logger.info(f"Directory already exists: {directory}")
        
        # Create subdirectories for specific purposes
        videos_upload_dir = upload_dir / "videos"
        videos_upload_dir.mkdir(parents=True, exist_ok=True)
        
        raw_videos_dir = upload_dir / "raw"
        raw_videos_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating application directories: {str(e)}")
        raise