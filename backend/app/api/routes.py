from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import json
import logging
from pathlib import Path
import shutil
import uuid
from datetime import datetime
import cv2
import base64
import asyncio
import time
import os

from ..core.config import settings
from ..core.websocket import manager
from ..models.schemas import (
    VideoInfo, DetectionInfo, IncidentInfo, CameraStatus,
    BillingActivity, CustomerData
)
from ..services.video_processor import video_processor
from database import (
    init_db, get_all_videos, get_detections, get_incidents,
    add_video, add_detection, add_incident, update_video_status,
    add_customer_data
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Ensure all directories exist before mounting them
for directory in [
    settings.UPLOAD_DIR,
    settings.PROCESSED_DIR,
    settings.FRAMES_DIR,
    settings.ALERTS_DIR,
    settings.THUMBNAILS_DIR
]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

# Mount static directories with error handling
try:
    router.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")
    router.mount("/processed", StaticFiles(directory=str(settings.PROCESSED_DIR)), name="processed")
    router.mount("/frames", StaticFiles(directory=str(settings.FRAMES_DIR)), name="frames")
    router.mount("/alerts", StaticFiles(directory=str(settings.ALERTS_DIR)), name="alerts")
    router.mount("/thumbnails", StaticFiles(directory=str(settings.THUMBNAILS_DIR)), name="thumbnails")
    logger.info("Static directories mounted successfully")
except Exception as e:
    logger.error(f"Error mounting static directories: {str(e)}")

@router.get("/")
async def root():
    return {"message": "Security Dashboard API is running"}

@router.get("/videos")
async def get_videos():
    """Get all videos and their status"""
    return await get_all_videos()


@router.post("/videos/face-extraction")
async def extract_faces(
    background_tasks: BackgroundTasks,
    video_path: str = Query(..., description="Path to input video"),
    confidence_threshold: float = Query(0.5, description="Confidence threshold for face detection")
):
    """Extract faces from video and save them"""
    try:
        # Create output directory
        save_path = settings.FRAMES_DIR / f"faces_{int(time.time())}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Start face extraction in background
        background_tasks.add_task(
            video_processor.process_face_extraction,
            video_path,
            str(save_path),
            confidence_threshold
        )
        
        return {
            "message": "Face extraction started",
            "save_path": str(save_path)
        }
    except Exception as e:
        logger.error(f"Error starting face extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/videos/loitering-detection")
async def detect_loitering(
    background_tasks: BackgroundTasks,
    video_path: str = Query(..., description="Path to input video"),
    threshold_time: int = Query(10, description="Time threshold for loitering in seconds")
):
    """
    Detect loitering in video with smooth tracking and persistent person identification.
    Analyzes how long people remain in camera view and creates alerts for extended stays.
    
    Args:
        video_path: Path to input video file
        threshold_time: Time threshold (in seconds) to consider loitering
        
    Returns:
        Processing job information
    """
    try:
        # Create output paths
        output_path = settings.PROCESSED_DIR / f"loitering_{Path(video_path).name}"
        screenshot_dir = settings.SCREENSHOTS_DIR / f"loitering_{int(time.time())}"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting loitering detection for video: {video_path}")
        logger.info(f"Screenshots will be saved to: {screenshot_dir}")
        logger.info(f"Using threshold time: {threshold_time}s")
        
        # Start loitering detection in background
        background_tasks.add_task(
            video_processor.process_loitering_detection_smooth,
            video_path,
            str(output_path),
            None,  # camera_id
            threshold_time
        )
        
        return {
            "message": "Loitering detection started with smooth tracking",
            "output_path": str(output_path),
            "screenshot_dir": str(screenshot_dir),
            "status": "processing",
            "threshold_time": threshold_time
        }
    except Exception as e:
        logger.error(f"Error starting loitering detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "start_stream":
                    camera_id = message.get("camera_id")
                    if camera_id:
                        # Start streaming for the specified camera
                        await get_live_camera_feed(camera_id)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {str(e)}")
                break
    finally:
        manager.disconnect(websocket)
        logger.info("WebSocket connection closed")

async def process_video_background(video_path: str, video_id: int, detection_type: str):
    """Process video in background and update status"""
    try:
        # Process video
        result = await video_processor.process_video(video_path, video_id, detection_type)
        
        # Update video status
        await update_video_status(video_id, "completed", result["output_path"])
        
        # Store detections
        for detection in result["detections"]:
            await add_detection({
                "video_id": video_id,
                "timestamp": datetime.now(),
                "frame_number": detection.get("frame_number", 0),
                "detection_type": detection["type"],
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "class_name": detection["class_name"],
                "image_path": detection.get("image_path", "")
            })
        
        # Broadcast completion message
        await manager.broadcast(json.dumps({
            "type": "processing_completed",
            "video_id": video_id,
            "output_path": result["output_path"],
            "thumbnail_path": result["thumbnail_path"]
        }))
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        await update_video_status(video_id, "failed")
        await manager.broadcast(json.dumps({
            "type": "processing_error",
            "video_id": video_id,
            "error": str(e)
        }))

async def store_detections(camera_id: int, detections: List[dict]):
    """Store detections in database"""
    try:
        for det in detections:
            await add_detection({
                "camera_id": camera_id,
                "timestamp": datetime.now(),
                "detection_type": det["type"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "class_name": det["class_name"],
                "frame_number": det.get("frame_number", 0)
            })
    except Exception as e:
        logger.error(f"Error storing detections: {str(e)}")
        raise

async def get_latest_detections():
    """Get latest detections from database"""
    try:
        detections = await get_detections()
        return [detection.to_dict() for detection in detections[:100]]  # Limit to 100 latest detections
    except Exception as e:
        logger.error(f"Error getting latest detections: {str(e)}")
        return [] 
    

@router.post("/videos/theft-detection")
async def detect_theft(
    background_tasks: BackgroundTasks,
    video_path: str = Query(..., description="Path to input video"),
    hand_stay_time_chest: float = Query(1.0, description="Time threshold for suspicious hand positions in chest area (seconds)"),
    hand_stay_time_waist: float = Query(1.5, description="Time threshold for suspicious hand positions in waist area (seconds)")
):
    """
    Detect suspicious behavior (potential theft) in video by monitoring hand movements.
    Uses improved tracking for smooth inference and consistent person IDs across frames.
    
    Args:
        video_path: Path to input video file
        hand_stay_time_chest: Threshold time (in seconds) for suspicious hand positions in chest area
        hand_stay_time_waist: Threshold time (in seconds) for suspicious hand positions in waist/pocket area
        
    Returns:
        Processing job information
    """
    try:
        # Create output paths
        output_path = settings.PROCESSED_DIR / f"theft_{Path(video_path).name}"
        screenshot_dir = settings.SCREENSHOTS_DIR / f"theft_{int(time.time())}"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting theft detection for video: {video_path}")
        logger.info(f"Screenshots will be saved to: {screenshot_dir}")
        logger.info(f"Using thresholds - chest: {hand_stay_time_chest}s, waist: {hand_stay_time_waist}s")
        
        # Temporarily override settings for this run
        original_chest_time = settings.HAND_STAY_TIME_CHEST
        original_waist_time = settings.HAND_STAY_TIME_WAIST
        
        settings.HAND_STAY_TIME_CHEST = hand_stay_time_chest
        settings.HAND_STAY_TIME_WAIST = hand_stay_time_waist
        
        # Start theft detection in background with smooth tracking
        background_tasks.add_task(
            process_with_settings_reset,
            video_processor.process_theft_detection_smooth,  # Use the smooth version with consistent tracking
            original_chest_time,
            original_waist_time,
            video_path,
            str(output_path),
            str(screenshot_dir)
        )
        
        return {
            "message": "Theft detection started with smooth tracking",
            "output_path": str(output_path),
            "screenshot_dir": str(screenshot_dir),
            "status": "processing",
            "thresholds": {
                "chest_area": hand_stay_time_chest,
                "waist_area": hand_stay_time_waist
            }
        }
    except Exception as e:
        logger.error(f"Error starting theft detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_with_settings_reset(detection_func, original_chest_time, original_waist_time, *args, **kwargs):
    """Helper function to process video and restore original settings afterward"""
    try:
        # Process the video
        result = await detection_func(*args, **kwargs)
        return result
    finally:
        # Restore original settings
        settings.HAND_STAY_TIME_CHEST = original_chest_time
        settings.HAND_STAY_TIME_WAIST = original_waist_time

async def process_with_settings_reset(detection_func, original_chest_time, original_waist_time, *args, **kwargs):
    """Helper function to process video and restore original settings afterward"""
    try:
        # Process the video
        result = await detection_func(*args, **kwargs)
        return result
    finally:
        # Restore original settings
        settings.HAND_STAY_TIME_CHEST = original_chest_time
        settings.HAND_STAY_TIME_WAIST = original_waist_time