import os
import logging

logger = logging.getLogger(__name__)

def ensure_app_directories(settings):
    """
    Ensure all necessary application directories exist
    
    Args:
        settings: Application settings object containing directory paths
    """
    logger.info("Initializing application directories")
    
    # List all directories that need to be created
    directories = [
        settings.DATA_DIR,
        settings.UPLOADS_DIR,  # Changed from UPLOAD_DIR to UPLOADS_DIR
        settings.PROCESSED_DIR,
        settings.THUMBNAILS_DIR,
        settings.SCREENSHOTS_DIR,
        settings.INCIDENTS_DIR,
        settings.MODELS_DIR
    ]
    
    # Create each directory if it doesn't exist
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")
            raise
    
    logger.info("All application directories initialized successfully")