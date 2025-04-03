from fastapi import (
    APIRouter, 
    HTTPException, 
    File, 
    UploadFile, 
    BackgroundTasks, 
    Query, 
    WebSocket, 
    WebSocketDisconnect,
    Depends
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict
import logging
import os
import uuid
import json
from datetime import datetime, timedelta
from fastapi import APIRouter

# Import existing routers
from .endpoints.video import router as video_router
from .endpoints.face_tracking import router as face_tracking_router
# Import our new detection router
from .endpoints.detection import router as detection_router
# Import core modules
from app.core.config import settings
# from app.core.websocket import websocket_manager

# Import services
from app.services.video_processor import video_processor
from app.services.face_recognition import face_recognition_service

# Import database operations
from database import (
    init_db, 
    get_all_videos, 
    get_detections, 
    get_incidents,
    add_video, 
    add_detection, 
    add_incident,
    update_video_status,
    get_customer_data,
    add_customer_data
)

# Import schemas
from app.models.schemas import (
    VideoInfo, 
    DetectionInfo, 
    IncidentInfo, 
    CameraStatus,
    CustomerData
)

# Setup logging
logger = logging.getLogger(__name__)

# Create main API router
router = APIRouter()

# Ensure upload directories exist
for directory in [
    settings.UPLOAD_DIR,
    settings.PROCESSED_DIR,
    settings.FRAMES_DIR,
    settings.ALERTS_DIR,
    settings.THUMBNAILS_DIR
]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

# Mount static file directories
router.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")
router.mount("/processed", StaticFiles(directory=str(settings.PROCESSED_DIR)), name="processed")
router.mount("/frames", StaticFiles(directory=str(settings.FRAMES_DIR)), name="frames")
router.mount("/alerts", StaticFiles(directory=str(settings.ALERTS_DIR)), name="alerts")
router.mount("/thumbnails", StaticFiles(directory=str(settings.THUMBNAILS_DIR)), name="thumbnails")
router.include_router(video_router, prefix="/videos", tags=["videos"])
router.include_router(detection_router, prefix="/detection", tags=["detection"])
# Root endpoint
@router.get("/")
async def root():
    """
    Root endpoint providing basic API information
    """
    return {
        "message": "Security Dashboard API",
        "version": "1.0.0",
        "status": "operational"
    }

# Video Processing Endpoints
@router.post("/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    detection_type: str = Query(..., 
        description="Type of detection (theft, loitering, face_detection)")
):
    """
    Upload and process video for detection
    
    Args:
        video (UploadFile): Video file to upload
        detection_type (str): Type of detection to perform
    
    Returns:
        Dict: Video processing job information
    """
    try:
        # Generate unique filename
        filename = f"{uuid.uuid4()}_{video.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await video.read())
        
        # Create video record in database
        video_id = await add_video({
            "name": video.filename,
            "file_path": file_path,
            "status": "pending",
            "detection_type": detection_type,
            "upload_time": datetime.now()
        })
        
        # Start video processing in background
        background_tasks.add_task(
            process_video_background, 
            file_path, 
            video_id, 
            detection_type
        )
        
        return {
            "message": "Video uploaded successfully",
            "video_id": video_id,
            "filename": filename,
            "detection_type": detection_type,
            "status": "processing"
        }
    
    except Exception as e:
        logger.error(f"Video upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos")
async def list_videos():
    """
    Retrieve list of all processed videos
    
    Returns:
        List[Dict]: Video information
    """
    try:
        return await get_all_videos()
    except Exception as e:
        logger.error(f"Error fetching videos: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving videos")

# Detection Endpoints
@router.get("/detections")
async def list_detections(
    video_id: Optional[int] = Query(None, description="Filter by video ID")
):
    """
    Retrieve detections with optional video ID filter
    
    Args:
        video_id (Optional[int]): Video ID to filter detections
    
    Returns:
        List[Dict]: Detection information
    """
    try:
        return await get_detections(video_id)
    except Exception as e:
        logger.error(f"Error fetching detections: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving detections")

# Incident Endpoints
@router.get("/incidents")
async def list_incidents(
    recent: bool = Query(False, description="Get only recent incidents")
):
    """
    Retrieve incidents with optional recent filter
    
    Args:
        recent (bool): Filter for recent incidents
    
    Returns:
        List[Dict]: Incident information
    """
    try:
        return await get_incidents(recent)
    except Exception as e:
        logger.error(f"Error fetching incidents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving incidents")

# Customer Data Endpoints
@router.get("/customers")
async def get_customers(
    gender: Optional[str] = Query(None),
    date: Optional[str] = Query(None)
):
    """
    Retrieve customer data with optional filters
    
    Args:
        gender (Optional[str]): Filter by gender
        date (Optional[str]): Filter by entry date
    
    Returns:
        List[Dict]: Customer data
    """
    try:
        filters = {}
        if gender:
            filters['gender'] = gender
        if date:
            filters['date'] = date
        
        return await get_customer_data(filters)
    except Exception as e:
        logger.error(f"Error fetching customer data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving customer data")

@router.post("/customers")
async def create_customer(customer_data: CustomerData):
    """
    Add new customer data
    
    Args:
        customer_data (CustomerData): Customer information
    
    Returns:
        Dict: Created customer information
    """
    try:
        customer_id = await add_customer_data(customer_data.dict())
        return {"message": "Customer added successfully", "customer_id": customer_id}
    except Exception as e:
        logger.error(f"Error adding customer: {str(e)}")
        raise HTTPException(status_code=500, detail="Error adding customer")

# Face Recognition Endpoints
@router.post("/face-recognition/track")
async def track_person_by_face(
    background_tasks: BackgroundTasks,
    face_image: UploadFile = File(...),
    video_path: Optional[str] = Query(None)
):
    """
    Track a person across video using face recognition
    
    Args:
        face_image (UploadFile): Face image to track
        video_path (Optional[str]): Optional video path to limit search
    
    Returns:
        Dict: Tracking job information
    """
    try:
        # Save uploaded face image
        face_filename = f"{uuid.uuid4()}_{face_image.filename}"
        face_path = os.path.join(settings.UPLOAD_DIR, "faces", face_filename)
        
        with open(face_path, "wb") as buffer:
            buffer.write(await face_image.read())
        
        # Start face tracking in background
        job_id = f"tracking_{uuid.uuid4()}"
        background_tasks.add_task(
            face_recognition_service.track_person, 
            face_path, 
            video_path, 
            job_id
        )
        
        return {
            "message": "Face tracking started",
            "job_id": job_id,
            "face_image": face_filename
        }
    
    except Exception as e:
        logger.error(f"Face tracking error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint (kept in a separate file for clarity)
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication
    
    Handles:
    - Connection management
    - Real-time updates
    - Streaming detection results
    """
    client_id = None
    
    

# Background video processing function
async def process_video_background(video_path: str, video_id: Optional[int], detection_type: str):
    """
    Process video in background and update status
    
    Args:
        video_path (str): Path to video file
        video_id (Optional[int]): Video database ID
        detection_type (str): Type of detection to perform
    """
    try:
        # Process video
        result = await video_processor.process_video(video_path, video_id, detection_type)
        
        # Update video status if ID is provided
        if video_id:
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
                "class_name": detection.get("class_name", "unknown")
            })
        
        # Broadcast completion
       
    
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        
        # Update video status to failed
        if video_id:
            await update_video_status(video_id, "failed")
        
        # Broadcast error
       

# System Health Check
@router.get("/health")
async def health_check():
    """
    Basic system health check endpoint
    
    Returns:
        Dict: System health status
    """
    try:
        # Add more comprehensive health checks as needed
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": "operational",
                "video_processing": "ready",
                "websocket": "active"
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="System health check failed")

# Include any additional routers or endpoints
# For example, face tracking, specific detection routes, etc.