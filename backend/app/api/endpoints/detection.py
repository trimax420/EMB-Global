from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from typing import Optional
import os
import aiofiles
from datetime import datetime
import uuid
import logging
import json
import pickle

from app.core.config import settings
from app.services.video_processor import video_processor
from database import add_detection, add_incident, update_video_status

router = APIRouter()
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
os.makedirs(settings.SCREENSHOTS_DIR, exist_ok=True)

@router.post("/loitering-detection")
async def detect_loitering(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    threshold_time: float = Query(10.0, description="Time threshold (in seconds) to consider loitering"),
    camera_id: Optional[int] = Query(None, description="Camera ID for identification")
):
    """
    Process a video for loitering detection.
    
    Args:
        video (UploadFile): Video file to analyze
        threshold_time (float): Time threshold to consider someone loitering (in seconds)
        camera_id (int, optional): Camera ID for identification
        
    Returns:
        Dict: Processing job information
    """
    try:
        # Validate video file
        if not video.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"loitering_{timestamp}_{uuid.uuid4().hex[:8]}_{video.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, "videos", filename)
        
        # Ensure directory exists
        os.makedirs(os.path.join(settings.UPLOAD_DIR, "videos"), exist_ok=True)
        
        # Save the uploaded video
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await video.read()
            await out_file.write(content)
        
        # Generate output paths
        output_filename = f"processed_loitering_{timestamp}.mp4"
        output_path = os.path.join(settings.PROCESSED_DIR, output_filename)
        
        # Create screenshots directory
        screenshots_dir = os.path.join(settings.SCREENSHOTS_DIR, f"loitering_{timestamp}")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Start processing in background
        job_id = f"loitering_{uuid.uuid4().hex[:8]}"
        background_tasks.add_task(
            process_loitering_detection,
            file_path,
            output_path,
            screenshots_dir,
            threshold_time,
            camera_id,
            job_id
        )
        
        return {
            "message": "Loitering detection started",
            "job_id": job_id,
            "video_path": file_path,
            "output_path": output_path,
            "screenshots_dir": screenshots_dir,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error starting loitering detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/theft-detection")
async def detect_theft(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    hand_stay_time: float = Query(2.0, description="Time threshold (in seconds) for suspicious hand positions"),
    camera_id: Optional[int] = Query(None, description="Camera ID for identification")
):
    """
    Process a video for theft detection.
    Args:
        video (UploadFile): Video file to analyze
        hand_stay_time (float): Time threshold for suspicious hand positions (in seconds)
        camera_id (int, optional): Camera ID for identification
        
    Returns:
        Dict: Processing job information
    """
    try:
        # Validate video file
        if not video.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"theft_{timestamp}_{uuid.uuid4().hex[:8]}_{video.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, "videos", filename)
        
        # Ensure directory exists
        os.makedirs(os.path.join(settings.UPLOAD_DIR, "videos"), exist_ok=True)
        
        # Save the uploaded video
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await video.read()
            await out_file.write(content)
        
        # Generate output paths
        output_filename = f"processed_theft_{timestamp}.mp4"
        output_path = os.path.join(settings.PROCESSED_DIR, output_filename)
        
        # Create screenshots directory
        screenshots_dir = os.path.join(settings.SCREENSHOTS_DIR, f"theft_{timestamp}")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Start processing in background
        job_id = f"theft_{uuid.uuid4().hex[:8]}"
        background_tasks.add_task(
            process_theft_detection,
            file_path,
            output_path,
            screenshots_dir,
            hand_stay_time,
            camera_id,
            job_id
        )
        
        return {
            "message": "Theft detection started",
            "job_id": job_id,
            "video_path": file_path,
            "output_path": output_path,
            "screenshots_dir": screenshots_dir,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error starting theft detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detection-results/{job_id}")
async def get_detection_results(job_id: str):
    """
    Get the results of a detection job.
    
    Args:
        job_id (str): ID of the job
        
    Returns:
        Dict: Detection results
    """
    try:
        # Check if results file exists
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        
        if not os.path.exists(results_path):
            return {
                "status": "processing",
                "message": "Detection results not available yet",
                "job_id": job_id
            }
        
        # Load results
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving detection results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_loitering_detection(
    video_path: str,
    output_path: str,
    screenshots_dir: str,
    threshold_time: float,
    camera_id: Optional[int],
    job_id: str
):
    """
    Process video for loitering detection in background
    
    Args:
        video_path (str): Path to video file
        output_path (str): Path to save processed video
        screenshots_dir (str): Directory to save detection screenshots
        threshold_time (float): Time threshold for loitering detection
        camera_id (int, optional): Camera ID for identification
        job_id (str): Unique job identifier
    """
    try:
        logger.info(f"Starting loitering detection job {job_id} for {video_path}")
        
        # Process video for loitering detection
        results = await video_processor.process_loitering_detection_smooth(
            video_path, 
            output_path, 
            camera_id, 
            threshold_time
        )
        
        # Add status to results
        results["status"] = "completed"
        results["job_id"] = job_id
        results["message"] = f"Loitering detection completed. Found {results.get('loitering_count', 0)} incidents."
        
        # Save results for later retrieval
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Loitering detection job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in loitering detection job {job_id}: {str(e)}")
        
        # Save error results
        results = {
            "status": "failed",
            "message": f"Error processing video: {str(e)}",
            "job_id": job_id
        }
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

async def process_theft_detection(
    video_path: str,
    output_path: str,
    screenshots_dir: str,
    hand_stay_time: float,
    camera_id: Optional[int],
    job_id: str
):
    """
    Process video for theft detection in background
    
    Args:
        video_path (str): Path to video file
        output_path (str): Path to save processed video
        screenshots_dir (str): Directory to save detection screenshots
        hand_stay_time (float): Time threshold for suspicious hand positions
        camera_id (int, optional): Camera ID for identification
        job_id (str): Unique job identifier
    """
    try:
        logger.info(f"Starting theft detection job {job_id} for {video_path}")
        
        # Process video for theft detection
        results = await video_processor.process_theft_detection_smooth(
            video_path, 
            output_path, 
            screenshots_dir, 
            hand_stay_time, 
            camera_id
        )
        
        # Add status to results
        results["status"] = "completed"
        results["job_id"] = job_id
        results["message"] = f"Theft detection completed. Found {results.get('total_theft_incidents', 0)} incidents."
        
        # Save results for later retrieval
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Theft detection job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in theft detection job {job_id}: {str(e)}")
        
        # Save error results
        results = {
            "status": "failed",
            "message": f"Error processing video: {str(e)}",
            "job_id": job_id
        }
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)