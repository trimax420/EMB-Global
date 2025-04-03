from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import os
import cv2
import logging
import json
from pathlib import Path
from ...core.config import settings
from ...services.ml_service import MLService
from ...services.video_processor import video_processor

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/debug-video-path")
async def debug_video_path(video_path: str = Query(...)):
    """
    Debug endpoint to check if a video path is valid and can be opened by OpenCV.
    
    Parameters:
        video_path: Path to the video file to check
        
    Returns:
        JSON response with debug information about the video path
    """
    response = {
        "original_path": video_path,
        "exists": False,
        "can_open": False,
        "normalized_path": None,
        "video_details": {},
        "error": None
    }
    
    try:
        # Check if the path exists directly
        if os.path.exists(video_path):
            response["exists"] = True
            response["normalized_path"] = os.path.abspath(video_path)
        else:
            # Try Windows-specific path normalization
            if os.name == 'nt':
                # Try backslashes
                backslash_path = video_path.replace('/', '\\')
                if os.path.exists(backslash_path):
                    response["exists"] = True
                    response["normalized_path"] = os.path.abspath(backslash_path)
                else:
                    # Try forward slashes
                    forward_slash_path = video_path.replace('\\', '/')
                    if os.path.exists(forward_slash_path):
                        response["exists"] = True
                        response["normalized_path"] = os.path.abspath(forward_slash_path)
        
        # Try uploads/raw directory as a fallback
        if not response["exists"]:
            uploads_dir = os.path.join(settings.UPLOAD_DIR, "raw")
            base_name = os.path.basename(video_path)
            fallback_path = os.path.join(uploads_dir, base_name)
            
            if os.path.exists(fallback_path):
                response["exists"] = True
                response["normalized_path"] = os.path.abspath(fallback_path)
                response["fallback_path_used"] = True
        
        # Check parent directory exists and list contents
        parent_dir = os.path.dirname(video_path)
        if os.path.exists(parent_dir):
            response["parent_dir_exists"] = True
            response["parent_dir_contents"] = os.listdir(parent_dir)[:20]  # Limit to 20 files
        
        # Try to open the video with OpenCV
        path_to_try = response["normalized_path"] if response["normalized_path"] else video_path
        cap = cv2.VideoCapture(path_to_try)
        if cap.isOpened():
            response["can_open"] = True
            # Get video details
            response["video_details"] = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
            }
            cap.release()
        
        # Add CWD and uploads directory for reference
        response["cwd"] = os.getcwd()
        response["upload_dir"] = settings.UPLOAD_DIR
        response["upload_dir_exists"] = os.path.exists(settings.UPLOAD_DIR)
        if response["upload_dir_exists"]:
            raw_dir = os.path.join(settings.UPLOAD_DIR, "raw")
            response["raw_dir_exists"] = os.path.exists(raw_dir)
            if response["raw_dir_exists"]:
                response["raw_dir_contents"] = os.listdir(raw_dir)[:20]  # Limit to 20 files
        
    except Exception as e:
        response["error"] = str(e)
        logger.error(f"Error debugging video path: {str(e)}")
    
    return response
    
@router.get("/check-video-source")
async def check_video_source(camera_id: Optional[int] = None, video_path: Optional[str] = None):
    """
    Check if a video source (camera or file) can be accessed.
    
    Parameters:
        camera_id: ID of the camera to check (optional)
        video_path: Path to a video file to check (optional)
        
    Returns:
        JSON response with information about the video source
    """
    if camera_id is None and video_path is None:
        raise HTTPException(status_code=400, detail="Either camera_id or video_path must be provided")
    
    response = {
        "source_type": "camera" if camera_id is not None else "file",
        "source_id": camera_id if camera_id is not None else video_path,
        "can_access": False,
        "details": {}
    }
    
    try:
        if video_path:
            # Apply normalization for Windows paths
            normalized_path = video_path
            if os.name == 'nt':
                if '/' in video_path:
                    normalized_path = video_path.replace('/', '\\')
            
            # First check if path exists
            if not os.path.exists(normalized_path):
                response["details"]["error"] = f"File not found: {normalized_path}"
                
                # Try in uploads/raw directory
                base_name = os.path.basename(normalized_path)
                fallback_path = os.path.join(settings.UPLOAD_DIR, "raw", base_name)
                if os.path.exists(fallback_path):
                    normalized_path = fallback_path
                    response["details"]["using_fallback_path"] = fallback_path
            
            cap = cv2.VideoCapture(normalized_path)
        else:
            cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            response["can_access"] = True
            response["details"] = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            }
            
            # Try to read a frame
            ret, frame = cap.read()
            response["details"]["can_read_frame"] = ret
            
            if ret:
                response["details"]["frame_shape"] = frame.shape
            
            cap.release()
        else:
            response["details"]["error"] = "Could not open video source"
            
    except Exception as e:
        response["details"]["error"] = str(e)
        logger.error(f"Error checking video source: {str(e)}")
    
    return response

# ... existing code ... 