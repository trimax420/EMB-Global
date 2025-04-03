from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
import requests
import tempfile
import pickle
import uuid
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.config import settings
from ..core.websocket import websocket_manager
from ..services.video_processor import video_processor

# Set up logging
logger = logging.getLogger(__name__)

# Define router - make sure this is at the top before any endpoint definitions
router = APIRouter()

# Helper function to download videos from URLs
async def download_from_url(url: str) -> str:
    """
    Download a file from a URL to a temporary file
    
    Args:
        url (str): URL to download
        
    Returns:
        str: Path to the downloaded temporary file
    """
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_path = temp_file.name
        temp_file.close()
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded video from {url} to {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Error downloading video from {url}: {str(e)}")
        raise

# Theft Detection Endpoint
@router.post("/videos/theft-detection")
async def process_theft_detection(
    background_tasks: BackgroundTasks,
    video_path: str = Query(..., description="Path or URL to the video file to process"),
    hand_stay_time_chest: float = Query(1.0, description="Time threshold for hand in chest area (seconds)"),
    hand_stay_time_waist: float = Query(1.5, description="Time threshold for hand in waist area (seconds)"),
    camera_id: Optional[int] = Query(None, description="Camera ID if available")
):
    """
    Process video for theft detection
    
    Args:
        video_path (str): Path or URL to video file
        hand_stay_time_chest (float): Time threshold for chest area
        hand_stay_time_waist (float): Time threshold for waist area
        camera_id (Optional[int]): Camera ID
        
    Returns:
        Dict: Processing job information
    """
    try:
        local_video_path = video_path
        
        # If the video path is a URL, download it first
        if video_path.startswith("http"):
            logger.info(f"Downloading video from URL: {video_path}")
            local_video_path = await download_from_url(video_path)
            logger.info(f"Video downloaded to: {local_video_path}")
        elif not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
            
        # Generate output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"theft_processed_{timestamp}_{Path(local_video_path).name}"
        output_path = os.path.join(settings.PROCESSED_DIR, output_filename)
        
        # Create screenshots directory
        screenshots_dir = os.path.join(settings.SCREENSHOTS_DIR, f"theft_{timestamp}")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Create tracking job ID
        job_id = f"theft_{uuid.uuid4().hex[:8]}"
        
        # Start theft detection in background
        background_tasks.add_task(
            process_theft_detection_background,
            local_video_path,
            output_path,
            screenshots_dir,
            hand_stay_time_chest,
            hand_stay_time_waist,
            camera_id,
            job_id
        )
        
        return {
            "message": "Theft detection processing started",
            "job_id": job_id,
            "video_path": video_path,  # Return original path for reference
            "local_video_path": local_video_path,  # Return local path for debugging
            "output_path": output_path,
            "screenshots_dir": screenshots_dir,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Theft detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Loitering Detection Endpoint
@router.post("/videos/loitering-detection")
async def process_loitering_detection(
    background_tasks: BackgroundTasks,
    video_path: str = Query(..., description="Path or URL to the video file to process"),
    threshold_time: float = Query(10.0, description="Time threshold for loitering detection (seconds)"),
    camera_id: Optional[int] = Query(None, description="Camera ID if available")
):
    """
    Process video for loitering detection
    
    Args:
        video_path (str): Path or URL to video file
        threshold_time (float): Time threshold for loitering
        camera_id (Optional[int]): Camera ID
        
    Returns:
        Dict: Processing job information
    """
    try:
        local_video_path = video_path
        
        # If the video path is a URL, download it first
        if video_path.startswith("http"):
            logger.info(f"Downloading video from URL: {video_path}")
            local_video_path = await download_from_url(video_path)
            logger.info(f"Video downloaded to: {local_video_path}")
        elif not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
            
        # Generate output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"loitering_processed_{timestamp}_{Path(local_video_path).name}"
        output_path = os.path.join(settings.PROCESSED_DIR, output_filename)
        
        # Create job ID
        job_id = f"loitering_{uuid.uuid4().hex[:8]}"
        
        # Start loitering detection in background
        background_tasks.add_task(
            process_loitering_detection_background,
            local_video_path,
            output_path,
            threshold_time,
            camera_id,
            job_id
        )
        
        return {
            "message": "Loitering detection processing started",
            "job_id": job_id,
            "video_path": video_path,  # Return original path for reference
            "local_video_path": local_video_path,  # Return local path for debugging
            "output_path": output_path,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Loitering detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Processing status endpoint
@router.get("/processing/status/{job_id}")
async def get_processing_status(
    job_id: str
):
    """
    Get the status of a processing job
    
    Args:
        job_id (str): Processing job ID
        
    Returns:
        Dict: Processing status information
    """
    try:
        # Check if results file exists
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        
        if not os.path.exists(results_path):
            return {
                "status": "processing",
                "message": "Processing in progress",
                "progress": 0
            }
        
        # Load results
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
            
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background processing functions
async def process_theft_detection_background(
    video_path: str,
    output_path: str,
    screenshots_dir: str,
    hand_stay_time_chest: float,
    hand_stay_time_waist: float,
    camera_id: Optional[int],
    job_id: str
):
    """
    Process theft detection in background
    """
    try:
        # Process the video
        results = await video_processor.process_theft_detection_smooth(
            video_path,
            output_path,
            screenshots_dir,
            hand_stay_time_chest,
            camera_id
        )
        
        # Add job ID to results
        results["job_id"] = job_id
        results["status"] = "completed"
        
        # Save results for later retrieval
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Broadcast completion via WebSocket
        await websocket_manager.broadcast({
            "type": "theft_detection_completed",
            "job_id": job_id,
            "total_incidents": results.get("total_theft_incidents", 0),
            "output_path": output_path
        })
        
        # Clean up temporary file if it was downloaded
        if video_path.startswith("/tmp/") and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"Removed temporary video file: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {video_path}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Theft detection background error: {str(e)}")
        
        # Save error results
        error_results = {
            "status": "failed",
            "message": f"Processing error: {str(e)}",
            "job_id": job_id
        }
        
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(error_results, f)
        
        # Broadcast error
        await websocket_manager.broadcast({
            "type": "theft_detection_failed",
            "job_id": job_id,
            "error": str(e)
        })
        
        # Clean up temporary file if it was downloaded
        if video_path.startswith("/tmp/") and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass

async def process_loitering_detection_background(
    video_path: str,
    output_path: str,
    threshold_time: float,
    camera_id: Optional[int],
    job_id: str
):
    """
    Process loitering detection in background
    """
    try:
        # Process the video
        results = await video_processor.process_loitering_detection_smooth(
            video_path,
            output_path,
            camera_id,
            threshold_time
        )
        
        # Add job ID to results
        results["job_id"] = job_id
        results["status"] = "completed"
        
        # Save results for later retrieval
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Broadcast completion via WebSocket
        await websocket_manager.broadcast({
            "type": "loitering_detection_completed",
            "job_id": job_id,
            "loitering_count": results.get("loitering_count", 0),
            "output_path": output_path
        })
        
        # Clean up temporary file if it was downloaded
        if video_path.startswith("/tmp/") and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"Removed temporary video file: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {video_path}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Loitering detection background error: {str(e)}")
        
        # Save error results
        error_results = {
            "status": "failed",
            "message": f"Processing error: {str(e)}",
            "job_id": job_id
        }
        
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(error_results, f)
        
        # Broadcast error
        await websocket_manager.broadcast({
            "type": "loitering_detection_failed",
            "job_id": job_id,
            "error": str(e)
        })
        
        # Clean up temporary file if it was downloaded
        if video_path.startswith("/tmp/") and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass