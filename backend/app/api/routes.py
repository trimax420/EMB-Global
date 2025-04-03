from fastapi import (
    APIRouter, 
    HTTPException, 
    File, 
    UploadFile, 
    BackgroundTasks, 
    Query, 
    WebSocket, 
    WebSocketDisconnect,
    Depends,
    Response,
    Body
)
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Union, Any
import logging
import os
import uuid
import json
from datetime import datetime, timedelta
import asyncio
import cv2
import random
from pathlib import Path
import traceback

# Import existing routers
from .endpoints.video import router as video_router
from .endpoints.face_tracking import router as face_tracking_router
# Import our new detection router
from .endpoints.detection import router as detection_router
# Import core modules
from ..core.config import settings

# Import our connection manager - fix the import path
from ..core.websocket import ConnectionManager

# Create a connection manager instance
manager = ConnectionManager()

# Import services - import the VideoProcessor class and the video_processor instance
from ..services.video_processor import video_processor, MLVideoStream
from ..services.face_recognition import face_recognition_service
from ..services.detection_service import DetectorService
from ..services.ml_service import MLService
from ..models.shared import BoundingBox
from ..models.camera import Camera, CreateCameraRequest

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
from ..models.schemas import (
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

# Create raw uploads directory if it doesn't exist
raw_uploads_dir = os.path.join(settings.UPLOAD_DIR, "raw")
os.makedirs(raw_uploads_dir, exist_ok=True)
logger.info(f"Ensured raw uploads directory exists: {raw_uploads_dir}")

# Mount static file directories
router.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")
router.mount("/uploads/raw", StaticFiles(directory=str(os.path.join(settings.UPLOAD_DIR, "raw"))), name="raw_uploads")
router.mount("/processed", StaticFiles(directory=str(settings.PROCESSED_DIR)), name="processed")
router.mount("/frames", StaticFiles(directory=str(settings.FRAMES_DIR)), name="frames")
router.mount("/alerts", StaticFiles(directory=str(settings.ALERTS_DIR)), name="alerts")
router.mount("/thumbnails", StaticFiles(directory=str(settings.THUMBNAILS_DIR)), name="thumbnails")
router.include_router(video_router, prefix="/videos", tags=["videos"])
router.include_router(detection_router, prefix="/detection", tags=["detection"])

# WebRTC endpoints
@router.post("/webrtc/offer")
async def webrtc_offer(
    offer: Dict = Body(...),
    camera_id: Optional[int] = Query(None, description="Camera ID to stream from"),
    video_path: Optional[str] = Query(None, description="Video file path to stream"),
    detection_type: str = Query("all", description="Type of detection to perform"),
    resolution: str = Query("original", description="Preferred resolution (e.g. '720p', '1080p', 'original')")
):
    """
    Handle WebRTC offer from client and start streaming video with ML processing
    
    Args:
        offer (Dict): SDP offer from client
        camera_id (Optional[int]): Camera ID to stream from
        video_path (Optional[str]): Video file path to stream
        detection_type (str): Type of detection to perform
        resolution (str): Preferred resolution for the video stream
        
    Returns:
        Dict: SDP answer to establish WebRTC connection
    """
    try:
        # Log the received video path for debugging
        logger.info(f"Received video path: {video_path}, resolution: {resolution}")
        
        # Ensure we have either camera_id or video_path
        if camera_id is None and not video_path:
            # For testing/demo purposes, use a default video if no source specified
            available_videos = await get_all_videos()
            if available_videos:
                video_path = available_videos[0].get("file_path")
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Must specify either camera_id or video_path"
                )
        
        # Check if the video path exists
        if video_path and not os.path.exists(video_path):
            logger.warning(f"Video path does not exist: {video_path}")
            # Try to find the file using a case-insensitive search in the uploads/raw directory
            raw_dir = os.path.join(settings.UPLOAD_DIR, "raw")
            if os.path.exists(raw_dir):
                # Get base filename
                base_name = os.path.basename(video_path)
                # Check if any file in raw_dir matches the base name (case-insensitive)
                for file in os.listdir(raw_dir):
                    if file.lower() == base_name.lower():
                        video_path = os.path.join(raw_dir, file)
                        logger.info(f"Found matching file: {video_path}")
                        break
        
        logger.info(f"Using video path for WebRTC: {video_path}")
        
        # Create WebRTC connection and return answer
        answer = await video_processor.create_rtc_connection(
            offer=offer,
            camera_id=camera_id,
            video_path=video_path,
            detection_type=detection_type,
            resolution=resolution
        )
        
        return JSONResponse(content=answer)
        
    except Exception as e:
        logger.error(f"WebRTC offer error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.on_event("shutdown")
async def on_shutdown():
    """Properly close all WebRTC connections when the server shuts down"""
    await video_processor.close_all_rtc_connections()

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
    - WebRTC signaling
    """
    client_id = str(uuid.uuid4())
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket client connected: {client_id}")
        
        # Send initial connection acknowledgment
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Handle messages
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_json()
                message_type = message.get("type", "")
                
                # Handle different message types
                if message_type == "ping":
                    # Simple ping-pong for connection keep-alive
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif message_type == "webrtc_offer":
                    # Process WebRTC offer
                    offer = message.get("offer")
                    camera_id = message.get("camera_id")
                    video_path = message.get("video_path")
                    detection_type = message.get("detection_type", "all")
                    resolution = message.get("resolution", "original")
                    
                    if not offer:
                        await websocket.send_json({
                            "type": "error",
                            "error": "Missing WebRTC offer"
                        })
                        continue
                    
                    # Create WebRTC connection
                    try:
                        answer = await video_processor.create_rtc_connection(
                            offer=offer,
                            camera_id=camera_id,
                            video_path=video_path,
                            detection_type=detection_type,
                            resolution=resolution
                        )
                        
                        # Send answer back to client
                        await websocket.send_json({
                            "type": "webrtc_answer",
                            "answer": answer
                        })
                    except Exception as e:
                        logger.error(f"WebRTC error: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "error": f"WebRTC error: {str(e)}"
                        })
                
                elif message_type == "request_detections":
                    # Client requesting latest detection results
                    video_id = message.get("video_id")
                    detections = await get_detections(video_id)
                    
                    await websocket.send_json({
                        "type": "detections_update",
                        "detections": detections
                    })
                    
                elif message_type == "request_incidents":
                    # Client requesting latest incidents
                    recent_only = message.get("recent", False)
                    incidents = await get_incidents(recent_only)
                    
                    await websocket.send_json({
                        "type": "incidents_update",
                        "incidents": incidents
                    })
                    
                else:
                    # Handle unknown message type
                    logger.warning(f"Unknown WebSocket message type: {message_type}")
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from client {client_id}")
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON message format"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up any resources for this client
        pass

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

@router.get("/video-file")
async def get_video_file(
    video_path: str = Query(..., description="Path to video file"),
):
    """
    Get direct information about a video file path for debugging purposes
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        Dict: Video file information
    """
    try:
        # Normalize the path
        normalized_path = os.path.normpath(video_path)
        
        # Check if the file exists
        exists = os.path.exists(normalized_path)
        
        # Get the parent directory
        parent_dir = os.path.dirname(normalized_path)
        
        # Check if the parent directory exists
        parent_exists = os.path.exists(parent_dir)
        
        # If the parent directory exists, list its contents
        dir_contents = []
        if parent_exists:
            try:
                dir_contents = os.listdir(parent_dir)
            except:
                dir_contents = ["Error listing directory"]
        
        # Get base name
        base_name = os.path.basename(normalized_path)
        
        # Look for possible matches in the raw uploads directory
        raw_dir = os.path.join(settings.UPLOAD_DIR, "raw")
        raw_dir_exists = os.path.exists(raw_dir)
        raw_contents = []
        if raw_dir_exists:
            try:
                raw_contents = os.listdir(raw_dir)
            except:
                raw_contents = ["Error listing directory"]
        
        return {
            "original_path": video_path,
            "normalized_path": normalized_path,
            "exists": exists,
            "parent_dir": parent_dir,
            "parent_exists": parent_exists,
            "dir_contents": dir_contents[:20],  # Limit to 20 files for brevity
            "base_name": base_name,
            "raw_dir": str(raw_dir),
            "raw_dir_exists": raw_dir_exists,
            "raw_contents": raw_contents[:20],  # Limit to 20 files for brevity
            "cwd": os.getcwd(),
            "upload_dir": str(settings.UPLOAD_DIR)
        }
    
    except Exception as e:
        logger.error(f"Video file check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoint section
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