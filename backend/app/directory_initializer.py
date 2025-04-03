import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_app_directories(settings):
    """
    Ensure all application directories exist.
    
    Args:
        settings: Application settings object with directory paths
    """
    directories = [
        settings.UPLOAD_DIR,
        settings.PROCESSED_DIR,
        settings.FRAMES_DIR,
        settings.ALERTS_DIR,
        settings.THUMBNAILS_DIR,
        settings.MODELS_DIR,
        settings.SCREENSHOTS_DIR
    ]
    
    # Create main directories - handle both Path and str objects
    for directory in directories:
        # Ensure directory exists
        if isinstance(directory, Path):
            directory.mkdir(parents=True, exist_ok=True)
            dir_str = str(directory)
        else:
            os.makedirs(directory, exist_ok=True)
            dir_str = directory
            
        logger.info(f"Ensured directory exists: {dir_str}")
    
    # Create subdirectories
    raw_uploads_dir = settings.UPLOAD_DIR / "raw" if isinstance(settings.UPLOAD_DIR, Path) else os.path.join(settings.UPLOAD_DIR, "raw")
    faces_dir = settings.UPLOAD_DIR / "faces" if isinstance(settings.UPLOAD_DIR, Path) else os.path.join(settings.UPLOAD_DIR, "faces")
    detection_results_dir = settings.PROCESSED_DIR / "detections" if isinstance(settings.PROCESSED_DIR, Path) else os.path.join(settings.PROCESSED_DIR, "detections")
    
    # Create these subdirectories
    if isinstance(raw_uploads_dir, Path):
        raw_uploads_dir.mkdir(parents=True, exist_ok=True)
    else:
        os.makedirs(raw_uploads_dir, exist_ok=True)
    logger.info(f"Ensured raw uploads directory exists: {raw_uploads_dir}")
    
    if isinstance(faces_dir, Path):
        faces_dir.mkdir(parents=True, exist_ok=True)
    else:
        os.makedirs(faces_dir, exist_ok=True)
    logger.info(f"Ensured faces directory exists: {faces_dir}")
    
    if isinstance(detection_results_dir, Path):
        detection_results_dir.mkdir(parents=True, exist_ok=True)
    else:
        os.makedirs(detection_results_dir, exist_ok=True)
    logger.info(f"Ensured detection results directory exists: {detection_results_dir}")
    
    # Print the path info for debugging
    logger.info(f"Working directory: {os.getcwd()}")
    abs_path = str(settings.UPLOAD_DIR.absolute()) if isinstance(settings.UPLOAD_DIR, Path) else os.path.abspath(settings.UPLOAD_DIR)
    logger.info(f"Upload directory absolute path: {abs_path}")
    
    # Check if the raw uploads directory contains sample videos
    raw_dir_str = str(raw_uploads_dir) if isinstance(raw_uploads_dir, Path) else raw_uploads_dir
    if os.path.exists(raw_dir_str):
        files = os.listdir(raw_dir_str)
        if files:
            logger.info(f"Found {len(files)} files in raw uploads directory: {files[:5]}")
        else:
            logger.warning(f"Raw uploads directory is empty: {raw_dir_str}")
            
    return True