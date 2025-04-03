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
    Request
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict
import logging
import os
import uuid
import json
import time
import asyncio
import cv2
from datetime import datetime, timedelta

# Import core modules
from app.core.config import settings
from app.core.websocket import websocket_manager

# Import services
from app.services.video_processor import video_processor

# Import detector
from detector import detector, active_detections, video_queue

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
    add_customer_data,
    get_cameras,
    update_camera_status,
    get_detection_stats,
    get_detections_by_camera
)

# Import schemas
from app.models.schemas import (
    VideoInfo, 
    DetectionInfo, 
    IncidentInfo, 
    CameraStatus,
    CustomerData
)

# Import the WebRTC router directly (the only one that exists)
from .webrtc import router as webrtc_router

# Setup logging
logger = logging.getLogger(__name__)

# Create a parent router that includes all the other routers
router = APIRouter()

# Include WebRTC router
router.include_router(webrtc_router, prefix="/webrtc", tags=["webrtc"])

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

@router.get("/detections/stats")
async def get_detection_statistics(
    camera_id: Optional[int] = Query(None, description="Filter by camera ID"),
    time_range: int = Query(24, description="Time range in hours")
):
    """
    Get detection statistics
    
    Args:
        camera_id (Optional[int]): Camera ID to filter by
        time_range (int): Time range in hours (default: 24)
        
    Returns:
        Dict: Detection statistics
    """
    try:
        # Get stats
        stats = await get_detection_stats(camera_id, time_range)
        
        return {
            "status": "success",
            "statistics": stats,
            "time_range": time_range,
            "camera_id": camera_id
        }
    except Exception as e:
        logger.error(f"Error getting detection statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cameras/{camera_id}/detections")
async def get_camera_detections(
    camera_id: int,
    limit: int = Query(100, description="Maximum number of detections to return")
):
    """
    Get detections for a specific camera
    
    Args:
        camera_id (int): Camera ID to get detections for
        limit (int): Maximum number of detections to return
        
    Returns:
        List[Dict]: Detections for the camera
    """
    try:
        # Get detections
        detections = await get_detections_by_camera(camera_id, limit)
        
        return {
            "status": "success",
            "camera_id": camera_id,
            "detections_count": len(detections),
            "detections": detections
        }
    except Exception as e:
        logger.error(f"Error getting camera detections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

# Loitering Detection Endpoint
@router.post("/detection/loitering")
async def detect_loitering(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    threshold_time: float = Query(10.0, description="Time threshold (seconds) to consider loitering"),
    camera_id: Optional[int] = Query(None, description="Camera ID for the video source")
):
    """
    Process video for loitering detection
    
    Args:
        video (UploadFile): Video file to analyze
        threshold_time (float): Time threshold in seconds to consider as loitering
        camera_id (Optional[int]): ID of the camera that captured the video
    
    Returns:
        Dict: Loitering detection job information
    """
    try:
        # Generate unique filename and paths
        filename = f"loitering_{uuid.uuid4()}_{video.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        output_path = os.path.join(settings.PROCESSED_DIR, f"loitering_result_{filename}")
        screenshots_dir = os.path.join(settings.SCREENSHOTS_DIR, f"loitering_{uuid.uuid4().hex[:8]}")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await video.read())
        
        # Create video record in database
        video_id = await add_video({
            "name": video.filename,
            "file_path": file_path,
            "status": "processing",
            "detection_type": "loitering",
            "upload_time": datetime.now()
        })
        
        # Start loitering detection in background
        job_id = f"loitering_{uuid.uuid4().hex[:8]}"
        
        # Use a wrapper function for the background task
        async def process_loitering_background():
            try:
                result = await video_processor.process_loitering_detection_smooth(
                    file_path, 
                    output_path,
                    camera_id, 
                    threshold_time
                )
                
                # Update video status
                await update_video_status(video_id, "completed", output_path)
                
                # Broadcast completion
                await websocket_manager.broadcast({
                    "type": "loitering_detection_completed",
                    "video_id": video_id,
                    "job_id": job_id,
                    "loitering_count": len(result.get("loitering_persons", [])),
                    "output_path": result.get("output_path")
                })
                
                return result
            except Exception as e:
                logger.error(f"Loitering detection error: {str(e)}")
                await update_video_status(video_id, "failed")
                await websocket_manager.broadcast({
                    "type": "loitering_detection_error",
                    "video_id": video_id,
                    "job_id": job_id,
                    "error": str(e)
                })
        
        background_tasks.add_task(process_loitering_background)
        
        return {
            "message": "Loitering detection started",
            "video_id": video_id,
            "job_id": job_id,
            "filename": filename,
            "status": "processing"
        }
    
    except Exception as e:
        logger.error(f"Loitering detection submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Theft Detection Endpoint
@router.post("/detection/theft")
async def detect_theft(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    hand_stay_time_chest: float = Query(2.0, description="Time threshold (seconds) for hand in chest area"),
    hand_stay_time_waist: float = Query(3.0, description="Time threshold (seconds) for hand in waist area"),
    camera_id: Optional[int] = Query(None, description="Camera ID for the video source")
):
    """
    Process video for theft detection
    
    Args:
        video (UploadFile): Video file to analyze
        hand_stay_time_chest (float): Time threshold (seconds) for hand in chest area
        hand_stay_time_waist (float): Time threshold (seconds) for hand in waist area
        camera_id (Optional[int]): ID of the camera that captured the video
    
    Returns:
        Dict: Theft detection job information
    """
    try:
        # Generate unique filename and paths
        filename = f"theft_{uuid.uuid4()}_{video.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        output_path = os.path.join(settings.PROCESSED_DIR, f"theft_result_{filename}")
        screenshots_dir = os.path.join(settings.SCREENSHOTS_DIR, f"theft_{uuid.uuid4().hex[:8]}")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await video.read())
        
        # Create video record in database
        video_id = await add_video({
            "name": video.filename,
            "file_path": file_path,
            "status": "processing",
            "detection_type": "theft",
            "upload_time": datetime.now()
        })
        
        # Start theft detection in background
        job_id = f"theft_{uuid.uuid4().hex[:8]}"
        
        # Use a wrapper function for the background task
        async def process_theft_background():
            try:
                # Store original settings
                original_chest_time = settings.HAND_STAY_TIME_CHEST
                original_waist_time = settings.HAND_STAY_TIME_WAIST
                
                # Set settings temporarily for this detection
                settings.HAND_STAY_TIME_CHEST = hand_stay_time_chest
                settings.HAND_STAY_TIME_WAIST = hand_stay_time_waist
                
                result = await video_processor.process_theft_detection_smooth(
                    file_path, 
                    output_path,
                    screenshots_dir,
                    hand_stay_time_chest,  # Pass the value directly
                    camera_id
                )
                
                # Restore original settings
                settings.HAND_STAY_TIME_CHEST = original_chest_time
                settings.HAND_STAY_TIME_WAIST = original_waist_time
                
                # Update video status
                await update_video_status(video_id, "completed", output_path)
                
                # Broadcast completion
                await websocket_manager.broadcast({
                    "type": "theft_detection_completed",
                    "video_id": video_id,
                    "job_id": job_id,
                    "theft_incidents": result.get("total_theft_incidents", 0),
                    "output_path": result.get("output_path")
                })
                
                return result
            except Exception as e:
                logger.error(f"Theft detection error: {str(e)}")
                await update_video_status(video_id, "failed")
                await websocket_manager.broadcast({
                    "type": "theft_detection_error",
                    "video_id": video_id,
                    "job_id": job_id,
                    "error": str(e)
                })
        
        background_tasks.add_task(process_theft_background)
        
        return {
            "message": "Theft detection started",
            "video_id": video_id,
            "job_id": job_id,
            "filename": filename,
            "status": "processing"
        }
    
    except Exception as e:
        logger.error(f"Theft detection submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints - using consistent naming
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for general real-time updates
    """
    client_id = None
    
    try:
        # Establish connection
        await websocket_manager.connect(websocket, client_id)
        
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle message types
                message_type = message.get('type')
                
                if message_type == 'connection_established':
                    client_id = message.get('client_id')
                    await websocket_manager.broadcast({
                        'type': 'connection_status',
                        'status': 'connected',
                        'client_id': client_id
                    })
                
                elif message_type == 'detection_request':
                    # Process detection request
                    video_path = message.get('video_path')
                    detection_type = message.get('detection_type')
                    
                    if video_path and detection_type:
                        # Start video processing
                        background_tasks = BackgroundTasks()
                        background_tasks.add_task(
                            process_video_background, 
                            video_path, 
                            None,  # No specific video ID 
                            detection_type
                        )
                
                # Add more message type handlers as needed
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await websocket.send_json({
                    'type': 'error',
                    'message': str(e)
                })
    
    finally:
        # Cleanup
        if client_id:
            websocket_manager.disconnect(websocket, client_id)

# WebSocket endpoint for real-time video inference
@router.websocket("/ws/inference")
async def inference_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time inference
    
    Handles streaming inference on video frames sent from the client
    """
    client_id = None
    active_inference = None
    tracking_context = None
    
    try:
        # Establish connection
        client_id = f"client_{uuid.uuid4().hex[:8]}"
        await websocket_manager.connect(websocket, client_id)
        logger.info(f"Inference WebSocket connection established: {client_id}")
        
        # Send initial connection confirmation
        await websocket.send_json({
            'type': 'connection_established',
            'client_id': client_id
        })
        
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_json()
                message_type = message.get('type', '')
                logger.debug(f"Received message type: {message_type} from client {client_id}")
                
                if message_type == 'start_inference':
                    # Stop any existing inference
                    if active_inference:
                        active_inference['active'] = False
                    
                    # Get inference parameters
                    detection_type = message.get('detection_type', 'all')
                    camera_id = message.get('camera_id')
                    source = message.get('source')  # 'webcam', 'rtsp', 'file', etc.
                    stream_url = message.get('stream_url')  # For RTSP/HTTP streams
                    
                    # Create inference context
                    active_inference = {
                        'active': True,
                        'detection_type': detection_type,
                        'camera_id': camera_id,
                        'source': source,
                        'stream_url': stream_url,
                        'frame_count': 0,
                        'start_time': time.time()
                    }
                    
                    # Initialize appropriate tracking context based on detection type
                    if detection_type == 'theft':
                        tracking_context = init_theft_detection_context(active_inference)
                        logger.info(f"Initialized theft tracking context for client {client_id}")
                    elif detection_type == 'loitering':
                        tracking_context = init_loitering_detection_context(active_inference)
                        logger.info(f"Initialized loitering tracking context for client {client_id}")
                    else:
                        tracking_context = None
                    
                    # Acknowledge start of inference
                    await websocket.send_json({
                        'type': 'inference_started',
                        'detection_type': detection_type,
                        'timestamp': datetime.now().isoformat()
                    })
                
                elif message_type == 'frame':
                    # Process a single frame sent by the client
                    if active_inference and active_inference['active']:
                        # Get frame data (base64 encoded)
                        frame_data = message.get('frame')
                        frame_number = message.get('frame_number', active_inference.get('frame_count', 0))
                        
                        if frame_data:
                            # Process frame based on detection type
                            detection_type = active_inference.get('detection_type', 'all')
                            
                            # Process the frame
                            # Decode base64 image
                            import base64
                            import cv2
                            import numpy as np
                            
                            try:
                                # Decode base64 image
                                _, encoded_data = frame_data.split(',', 1)
                                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if frame is None:
                                    raise ValueError("Could not decode frame")
                                
                                # Process based on detection type
                                if detection_type == 'theft':
                                    results, annotated_frame = await process_theft_frame(
                                        frame, active_inference, tracking_context
                                    )
                                elif detection_type == 'loitering':
                                    results, annotated_frame = await process_loitering_frame(
                                        frame, active_inference, tracking_context
                                    )
                                else:
                                    # Default to regular object detection
                                    results = await video_processor.process_frame(frame, detection_type='all')
                                    annotated_frame = annotate_frame(frame, results, detection_type)
                                
                                # Update frame count
                                active_inference['frame_count'] = frame_number + 1
                                
                                # Encode the processed frame
                                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                encoded_frame = base64.b64encode(buffer).decode('utf-8')
                                
                                # Send the result
                                await websocket.send_json({
                                    'type': 'inference_result',
                                    'frame_number': frame_number,
                                    'detection_type': detection_type,
                                    'detections': results,
                                    'processed_frame': f"data:image/jpeg;base64,{encoded_frame}"
                                })
                                
                                # Log detections if any
                                if results and len(results) > 0:
                                    logger.info(f"Client {client_id}: {len(results)} detections in frame {frame_number}")
                                
                            except Exception as process_error:
                                logger.error(f"Error processing frame: {str(process_error)}")
                                logger.error(traceback.format_exc())
                                await websocket.send_json({
                                    'type': 'frame_error',
                                    'frame_number': frame_number,
                                    'error': str(process_error)
                                })
                
                elif message_type == 'stop_inference':
                    # Stop active inference
                    if active_inference:
                        active_inference['active'] = False
                        await websocket.send_json({
                            'type': 'inference_stopped',
                            'frames_processed': active_inference.get('frame_count', 0),
                            'duration_seconds': time.time() - active_inference.get('start_time', time.time())
                        })
                        active_inference = None
                        tracking_context = None
                
            except WebSocketDisconnect:
                logger.info(f"Inference WebSocket disconnected: {client_id}")
                break
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id}")
                await websocket.send_json({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                })
                
            except Exception as e:
                logger.error(f"Inference WebSocket error: {str(e)}")
                logger.error(traceback.format_exc())
                await websocket.send_json({
                    'type': 'error',
                    'message': str(e)
                })
                
    finally:
        # Clean up resources
        if client_id:
            websocket_manager.disconnect(websocket, client_id)
        if active_inference:
            active_inference['active'] = False
        logger.info(f"Inference WebSocket connection closed: {client_id}")

async def process_theft_frame(frame, inference_context, tracking_context=None):
    """Process frame for theft detection using existing theft detection logic"""
    try:
        if tracking_context is None:
            tracking_context = init_theft_detection_context(inference_context)
            
        # Use the existing video processor for pose detection
        logger.debug("Processing frame for theft detection")
        
        # Update tracking context time
        current_time = time.time()
        if 'last_frame_time' in tracking_context:
            frame_time = current_time - tracking_context['last_frame_time']
        else:
            frame_time = 0.03  # Assume ~30 fps if first frame
        tracking_context['last_frame_time'] = current_time
        tracking_context['frame_count'] = tracking_context.get('frame_count', 0) + 1
        
        # Process frame using pose detection to find people and keypoints
        pose_results = await video_processor.process_frame(frame, detection_type='pose')
        
        if not pose_results:
            logger.debug("No pose detections found in frame")
            return [], frame
        
        # Create a temporary directory for any screenshots if needed
        if 'screenshots_dir' not in tracking_context:
            temp_dir = os.path.join(settings.SCREENSHOTS_DIR, f"realtime_theft_{uuid.uuid4().hex[:8]}")
            os.makedirs(temp_dir, exist_ok=True)
            tracking_context['screenshots_dir'] = temp_dir
            
        # Extract keypoints and analyze hands
        results = []
        annotated_frame = frame.copy()
        
        # Get relevant keypoint indices from video_processor
        LEFT_WRIST = video_processor.LEFT_WRIST
        RIGHT_WRIST = video_processor.RIGHT_WRIST
        NOSE = video_processor.NOSE
        
        for detection in pose_results:
            if detection['type'] == 'pose' and 'keypoints' in detection:
                keypoints = detection['keypoints']
                bbox = detection['bbox']
                
                # Get relevant parts
                left_wrist = keypoints[LEFT_WRIST] if LEFT_WRIST < len(keypoints) and keypoints[LEFT_WRIST][0] > 0 else None
                right_wrist = keypoints[RIGHT_WRIST] if RIGHT_WRIST < len(keypoints) and keypoints[RIGHT_WRIST][0] > 0 else None
                nose = keypoints[NOSE] if NOSE < len(keypoints) and keypoints[NOSE][0] > 0 else None
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, bbox)
                width, height = x2 - x1, y2 - y1
                
                # Define detection zones (same as in theft detection)
                chest_box = [x1 + int(0.1 * width), y1, x2 - int(0.1 * width), y1 + int(0.4 * height)]
                left_waist_box = [x1, y1 + int(0.5 * height), x1 + int(0.5 * width), y2]
                right_waist_box = [x1 + int(0.5 * width), y1 + int(0.5 * height), x2, y2]
                
                # Draw detection zones for visualization
                cv2.rectangle(annotated_frame, tuple(chest_box[:2]), tuple(chest_box[2:]), (0, 255, 255), 2)
                cv2.rectangle(annotated_frame, tuple(left_waist_box[:2]), tuple(left_waist_box[2:]), (255, 0, 0), 2)
                cv2.rectangle(annotated_frame, tuple(right_waist_box[:2]), tuple(right_waist_box[2:]), (0, 0, 255), 2)
                
                # Draw skeleton for visualization
                for i in range(len(keypoints)):
                    if keypoints[i][0] > 0 and keypoints[i][1] > 0:
                        cv2.circle(annotated_frame, (int(keypoints[i][0]), int(keypoints[i][1])), 5, (0, 255, 0), -1)
                
                # Get a unique ID for this person
                person_id = f"person_{int(x1/10)}_{int(y1/10)}"
                
                # Track hand positions
                for wrist, hand_label in [(left_wrist, f"{person_id}_left"), (right_wrist, f"{person_id}_right")]:
                    if wrist is None:
                        continue
                        
                    wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                    wrist_box = [wrist_x - 10, wrist_y - 10, wrist_x + 10, wrist_y + 10]
                    
                    # Check if hand is in a suspicious zone
                    def is_intersecting(box1, box2):
                        return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])
                        
                    in_chest = is_intersecting(wrist_box, chest_box)
                    in_left_waist = is_intersecting(wrist_box, left_waist_box)
                    in_right_waist = is_intersecting(wrist_box, right_waist_box)
                    
                    # Initialize tracking for this hand if not already tracked
                    if 'hand_positions' not in tracking_context:
                        tracking_context['hand_positions'] = {}
                        
                    if hand_label not in tracking_context['hand_positions']:
                        tracking_context['hand_positions'][hand_label] = []
                    
                    # Record current hand position and time
                    if in_chest:
                        zone = "chest"
                    elif in_left_waist:
                        zone = "left_waist"
                    elif in_right_waist:
                        zone = "right_waist"
                    else:
                        zone = "other"
                    
                    # Add to hand position history
                    current_frame_time = tracking_context['frame_count'] / 30.0  # Estimate time at 30fps
                    tracking_context['hand_positions'][hand_label].append((current_frame_time, (wrist_x, wrist_y), zone))
                    
                    # Limit history size
                    if len(tracking_context['hand_positions'][hand_label]) > 60:  # Keep ~2 seconds
                        tracking_context['hand_positions'][hand_label].pop(0)
                    
                    # Analyze hand positions for theft detection
                    if in_chest or in_left_waist or in_right_waist:
                        # Determine which zone and appropriate threshold
                        zone_type = "chest" if in_chest else "waist"
                        threshold = tracking_context.get('threshold_chest', 2.0) if in_chest else tracking_context.get('threshold_waist', 3.0)
                        
                        # Count how long the hand has been in this zone continuously
                        zone_duration = 0
                        continuous_in_zone = True
                        
                        # Check hand position history in reverse order
                        for idx in range(len(tracking_context['hand_positions'][hand_label]) - 1, 0, -1):
                            past_time, _, past_zone = tracking_context['hand_positions'][hand_label][idx]
                            prev_time, _, prev_zone = tracking_context['hand_positions'][hand_label][idx - 1]
                            
                            # If zone changes, break the continuity
                            if past_zone != zone and past_zone != "other":
                                continuous_in_zone = False
                                break
                            
                            # If zone matches, add to duration
                            if past_zone == zone:
                                zone_duration += (past_time - prev_time)
                        
                        # Check if duration exceeds threshold
                        if zone_duration >= threshold:
                            # Generate theft detection result
                            theft_result = {
                                "type": "theft",
                                "person_id": person_id,
                                "confidence": 0.8,
                                "bbox": bbox,
                                "zone": zone_type,
                                "frame_number": tracking_context['frame_count'],
                                "duration": zone_duration
                            }
                            results.append(theft_result)
                            
                            # Highlight suspicious hand on visualization
                            cv2.rectangle(annotated_frame, 
                                        (wrist_box[0], wrist_box[1]), 
                                        (wrist_box[2], wrist_box[3]), 
                                        (0, 0, 255), -1)
                            
                            # Add text label
                            cv2.putText(annotated_frame, 
                                       f"THEFT: Hand in {zone_type}", 
                                       (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            logger.info(f"Theft detection: Hand in {zone_type} for {zone_duration:.2f}s")
        
        return results, annotated_frame
        
    except Exception as e:
        logger.error(f"Error in real-time theft detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [], frame

async def process_loitering_frame(frame, inference_context, tracking_context=None):
    """Process frame for loitering detection using existing loitering detection logic"""
    try:
        if tracking_context is None:
            tracking_context = init_loitering_detection_context(inference_context)
            
        logger.debug("Processing frame for loitering detection")
        
        # Update tracking context time
        current_time = time.time()
        if 'last_frame_time' in tracking_context:
            frame_time = current_time - tracking_context['last_frame_time']
        else:
            frame_time = 0.03  # Assume ~30 fps if first frame
        tracking_context['last_frame_time'] = current_time
        tracking_context['frame_count'] = tracking_context.get('frame_count', 0) + 1
        
        # Create a structure for tracking people similar to loitering detection
        if 'tracked_persons' not in tracking_context:
            tracking_context['tracked_persons'] = {}
            
        # Create a temporary directory for any screenshots if needed
        if 'screenshots_dir' not in tracking_context:
            temp_dir = os.path.join(settings.SCREENSHOTS_DIR, f"realtime_loitering_{uuid.uuid4().hex[:8]}")
            os.makedirs(temp_dir, exist_ok=True)
            tracking_context['screenshots_dir'] = temp_dir
        
        # Detect people
        object_results = await video_processor.process_frame(frame, detection_type='object')
        
        # Filter for people
        people_detections = [d for d in object_results if d.get('class_name') == 'person']
        
        if not people_detections:
            logger.debug("No people detected in frame")
            return [], frame
        
        # Initialize results
        results = []
        annotated_frame = frame.copy()
        
        # Helper function to extract features for tracking
        def extract_person_features(frame, bbox):
            """Extract color histogram features from person region for re-identification"""
            import cv2
            import numpy as np
            
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure bbox is within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None  # Invalid bbox
                
            person_roi = frame[y1:y2, x1:x2]
            # Convert to HSV colorspace for better feature representation
            hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
            # Calculate histogram
            hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
            # Normalize the histogram
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist.flatten()

        # Process each person detection
        for detection in people_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Extract person features for tracking
            features = extract_person_features(frame, bbox)
            if features is None:
                continue
                
            # Calculate center of detection
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Try to match with existing tracked persons
            best_match_id = None
            best_match_score = 0
            
            for person_id, person_data in tracking_context['tracked_persons'].items():
                # Skip if not seen recently
                if 'last_seen_time' not in person_data:
                    continue
                    
                # Skip if not seen in last 2 seconds
                if current_time - person_data['last_seen_time'] > 2.0:
                    continue
                
                # Calculate IoU between current bbox and tracked person's bbox
                prev_bbox = person_data['bbox']
                
                # Calculate feature similarity
                if 'features' in person_data and person_data['features'] is not None:
                    import cv2
                    feature_similarity = cv2.compareHist(
                        features.reshape(-1, 1),
                        person_data['features'].reshape(-1, 1),
                        cv2.HISTCMP_CORREL
                    )
                    
                    if feature_similarity > best_match_score and feature_similarity > 0.5:
                        best_match_score = feature_similarity
                        best_match_id = person_id
            
            # Assign or create person ID
            if best_match_id:
                person_id = best_match_id
                # Update tracking info
                tracking_context['tracked_persons'][person_id].update({
                    'bbox': bbox,
                    'features': features,
                    'position': [center_x, center_y],
                    'last_seen_time': current_time,
                })
                
                # Calculate time since first seen
                time_present = current_time - tracking_context['tracked_persons'][person_id]['first_seen_time']
                tracking_context['tracked_persons'][person_id]['time_present'] = time_present
                
            else:
                # New person
                person_id = f"person_{len(tracking_context['tracked_persons']) + 1}"
                
                tracking_context['tracked_persons'][person_id] = {
                    'bbox': bbox,
                    'features': features,
                    'position': [center_x, center_y],
                    'first_seen_time': current_time,
                    'last_seen_time': current_time,
                    'time_present': 0.0,
                }
            
            # Check for loitering based on time present
            time_present = tracking_context['tracked_persons'][person_id].get('time_present', 0)
            threshold_time = tracking_context.get('threshold_time', 10.0)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            
            if time_present >= threshold_time:
                # Person is loitering - red box
                color = (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, 
                           f"{person_id}: {time_present:.1f}s", 
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Create loitering result
                loitering_result = {
                    "type": "loitering",
                    "person_id": person_id,
                    "confidence": confidence,
                    "bbox": bbox,
                    "time_present": time_present,
                    "frame_number": tracking_context['frame_count']
                }
                results.append(loitering_result)
                logger.info(f"Loitering detected: Person {person_id} present for {time_present:.2f}s")
                
            else:
                # Normal tracking - green box
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, 
                           f"{person_id}: {time_present:.1f}s", 
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Clean up old tracked persons
        for person_id in list(tracking_context['tracked_persons'].keys()):
            if current_time - tracking_context['tracked_persons'][person_id]['last_seen_time'] > 5.0:
                del tracking_context['tracked_persons'][person_id]
        
        return results, annotated_frame
        
    except Exception as e:
        logger.error(f"Error in real-time loitering detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [], frame

# Helper function implementations for detection contexts
def init_theft_detection_context(inference_context):
    """Initialize context for theft detection tracking"""
    logger.info("Initializing theft detection context")
    return {
        'tracked_persons': {},  # Track people across frames
        'hand_positions': {},   # Track hand positions over time
        'frame_count': 0,
        'last_frame_time': time.time(),
        'detection_type': 'theft',
        'threshold_chest': 2.0,  # Time threshold for hands in chest area
        'threshold_waist': 3.0   # Time threshold for hands in waist area
    }

def init_loitering_detection_context(inference_context):
    """Initialize context for loitering detection tracking"""
    threshold = inference_context.get('threshold_time', 10.0)
    logger.info(f"Initializing loitering detection context with threshold {threshold}s")
    return {
        'tracked_persons': {},  # Track people across frames
        'frame_count': 0,
        'last_frame_time': time.time(),
        'detection_type': 'loitering',
        'threshold_time': threshold,  # Time threshold for loitering
    }

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
        await websocket_manager.broadcast({
            "type": "processing_completed",
            "video_id": video_id,
            "results": result
        })
    
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        
        # Update video status to failed
        if video_id:
            await update_video_status(video_id, "failed")
        
        # Broadcast error
        await websocket_manager.broadcast({
            "type": "processing_error",
            "video_id": video_id,
            "error": str(e)
        })

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

# Live Video Streaming Endpoints
@router.get("/cameras")
async def get_available_cameras():
    """
    Get all available camera feeds
    
    Returns:
        List[Dict]: Camera information including stream URLs
    """
    try:
        # Get cameras from database
        db_cameras = await get_cameras()
        cameras = list(db_cameras) if db_cameras else []
        
        # Also add videos from the public videos directory
        if settings.PUBLIC_VIDEOS_DIR.exists():
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []
            
            # List all video files in the public videos directory
            for file in settings.PUBLIC_VIDEOS_DIR.glob('**/*'):
                if file.is_file() and file.suffix.lower() in video_extensions:
                    video_files.append(file)
            
            # Create a camera entry for each video file
            for i, video_file in enumerate(video_files, start=len(cameras) + 1):
                # Extract video name from path
                video_name = video_file.stem.replace('_', ' ').title()
                
                # Add to cameras list
                cameras.append({
                    "id": i,
                    "name": f"Video Feed: {video_name}",
                    "status": "available",
                    "stream_url": str(video_file),
                    "is_streaming": False,
                    "video_path": str(video_file),
                    "type": "file"
                })
            
            logger.info(f"Found {len(video_files)} video files in public directory")
        
        # Add active status information from detector
        for camera in cameras:
            camera_id = camera.get("id")
            camera["is_streaming"] = camera_id in detector.active_streams
            if camera["is_streaming"]:
                camera["stream_url"] = f"/api/cameras/{camera_id}/live"
                camera["stream_since"] = detector.active_streams[camera_id]["start_time"].isoformat()
            
        return cameras
    except Exception as e:
        logger.error(f"Error fetching cameras: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cameras/{camera_id}/stream/start")
async def start_camera_stream(
    camera_id: int, 
    video_path: Optional[str] = Query(None, description="Optional video path for file-based streaming")
):
    """
    Start live video stream for a camera
    
    Args:
        camera_id (int): Camera ID to stream
        video_path (str, optional): Path to video file for simulation
        
    Returns:
        Dict: Stream status information
    """
    try:
        # Get all cameras including file-based ones
        cameras = await get_available_cameras()  # Use the function that includes public videos
        camera = next((c for c in cameras if c["id"] == camera_id), None)
        
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Check if already streaming
        if camera_id in detector.active_streams:
            return {"status": "warning", "message": f"Camera {camera_id} is already streaming"}
        
        # Get video path (use provided or from camera config)
        stream_path = video_path or camera.get("video_path") or camera.get("stream_url")
        if not stream_path:
            raise HTTPException(status_code=400, detail="No video path provided and no stream URL configured for camera")
        
        logger.info(f"Starting camera stream for camera {camera_id} using path: {stream_path}")
        
        # Start the stream
        result = await detector.start_live_stream(camera_id, stream_path)
        
        # Update camera status in database (if it's a DB-based camera)
        try:
            if camera.get("type") != "file":
                await update_camera_status(camera_id, "online")
        except Exception as e:
            logger.warning(f"Failed to update camera status: {str(e)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting camera stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cameras/{camera_id}/stream/stop")
async def stop_camera_stream(camera_id: int):
    """
    Stop live video stream for a camera
    
    Args:
        camera_id (int): Camera ID to stop streaming
        
    Returns:
        Dict: Stream status information
    """
    try:
        result = await detector.stop_live_stream(camera_id)
        
        # Update camera status
        await update_camera_status(camera_id, "offline")
        
        return result
        
    except Exception as e:
        logger.error(f"Error stopping camera stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cameras/{camera_id}/detections")
async def get_camera_detections(
    camera_id: int,
    limit: int = Query(100, description="Maximum number of detections to return")
):
    """
    Get detections for a specific camera
    
    Args:
        camera_id (int): Camera ID to get detections for
        limit (int): Maximum number of detections to return
        
    Returns:
        List[Dict]: Detections for the camera
    """
    try:
        # Get detections
        detections = await get_detections_by_camera(camera_id, limit)
        
        return {
            "status": "success",
            "camera_id": camera_id,
            "detections_count": len(detections),
            "detections": detections
        }
    except Exception as e:
        logger.error(f"Error getting camera detections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for real-time video and detection streaming
@router.websocket("/ws/live")
async def live_detection_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming with detections
    
    The client can send messages to control the stream:
    - {"action": "subscribe", "camera_id": 1} to subscribe to a specific camera
    - {"action": "unsubscribe", "camera_id": 1} to unsubscribe from a camera
    - {"action": "list_cameras"} to get a list of available cameras
    """
    client_id = None
    try:
        # Connect to WebSocket
        client_id = await websocket_manager.connect(websocket)
        logger.info(f"Client {client_id} connected to live detection WebSocket")
        
        # Track subscribed cameras for this client
        subscribed_cameras = set()
        
        # Send initial camera list
        cameras = await get_cameras()
        await websocket.send_json({
            "type": "camera_list",
            "cameras": [
                {
                    "id": camera["id"],
                    "name": camera["name"],
                    "status": camera["status"],
                    "is_streaming": camera["id"] in detector.active_streams
                } for camera in cameras
            ]
        })
        
        # Handle client messages
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_text()
                data = json.loads(message)
                action = data.get("action")
                
                if action == "subscribe":
                    camera_id = data.get("camera_id")
                    if camera_id:
                        # Check if camera exists
                        cameras = await get_cameras()
                        camera = next((c for c in cameras if c["id"] == camera_id), None)
                        
                        if not camera:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Camera {camera_id} not found"
                            })
                            continue
                        
                        # Check if camera is streaming
                        if camera_id not in detector.active_streams:
                            # Auto-start the stream if not already running
                            stream_path = camera.get("stream_url")
                            if stream_path:
                                await detector.start_live_stream(camera_id, stream_path)
                                await update_camera_status(camera_id, "online")
                                await websocket.send_json({
                                    "type": "stream_started",
                                    "camera_id": camera_id
                                })
                            else:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Camera {camera_id} has no stream URL configured"
                                })
                                continue
                        
                        # Add to subscribed cameras
                        subscribed_cameras.add(camera_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "camera_id": camera_id
                        })
                
                elif action == "unsubscribe":
                    camera_id = data.get("camera_id")
                    if camera_id and camera_id in subscribed_cameras:
                        subscribed_cameras.remove(camera_id)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "camera_id": camera_id
                        })
                
                elif action == "list_cameras":
                    cameras = await get_cameras()
                    await websocket.send_json({
                        "type": "camera_list",
                        "cameras": [
                            {
                                "id": camera["id"],
                                "name": camera["name"],
                                "status": camera["status"],
                                "is_streaming": camera["id"] in detector.active_streams
                            } for camera in cameras
                        ]
                    })
                
                elif action == "start_stream":
                    camera_id = data.get("camera_id")
                    if camera_id:
                        cameras = await get_cameras()
                        camera = next((c for c in cameras if c["id"] == camera_id), None)
                        
                        if not camera:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Camera {camera_id} not found"
                            })
                            continue
                        
                        stream_path = camera.get("stream_url")
                        if not stream_path:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Camera {camera_id} has no stream URL configured"
                            })
                            continue
                        
                        # Start the stream
                        await detector.start_live_stream(camera_id, stream_path)
                        await update_camera_status(camera_id, "online")
                        
                        # Auto-subscribe
                        subscribed_cameras.add(camera_id)
                        
                        await websocket.send_json({
                            "type": "stream_started",
                            "camera_id": camera_id
                        })
                
                elif action == "stop_stream":
                    camera_id = data.get("camera_id")
                    if camera_id:
                        result = await detector.stop_live_stream(camera_id)
                        await update_camera_status(camera_id, "offline")
                        
                        # Auto-unsubscribe
                        if camera_id in subscribed_cameras:
                            subscribed_cameras.remove(camera_id)
                        
                        await websocket.send_json({
                            "type": "stream_stopped",
                            "camera_id": camera_id
                        })
            
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message"
                })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected from live detection WebSocket")
    except Exception as e:
        logger.error(f"Error in live detection WebSocket: {str(e)}")
    finally:
        if client_id:
            websocket_manager.disconnect(websocket, client_id)

# Modified video upload endpoint to use the new detector
@router.post("/videos/upload-with-realtime")
async def upload_video_with_realtime(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    detection_type: str = Query(..., description="Type of detection to perform")
):
    """
    Upload and process video with real-time updates via WebSocket
    
    Args:
        video (UploadFile): Video file to upload
        detection_type (str): Type of detection to perform
    
    Returns:
        Dict: Video processing information
    """
    try:
        # Generate unique filename
        filename = f"{uuid.uuid4()}_{video.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await video.read())
        
        # Generate thumbnail
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        thumbnail_path = None
        if ret:
            thumbnail_path = os.path.join(settings.THUMBNAILS_DIR, f"thumb_{filename}.jpg")
            cv2.imwrite(thumbnail_path, frame)
        cap.release()
        
        # Create video record in database
        video_id = await add_video({
            "name": video.filename,
            "file_path": file_path,
            "thumbnail_path": thumbnail_path,
            "status": "processing",
            "detection_type": detection_type,
            "upload_time": datetime.now()
        })
        
        # Define output path
        output_path = os.path.join(settings.PROCESSED_DIR, f"processed_{video_id}.mp4")
        
        # Start video processing in background with WebSocket updates
        background_tasks.add_task(
            detector.process_video_with_customer_data,
            file_path,
            video_id,
            detection_type,
            output_path
        )
        
        return {
            "message": "Video uploaded successfully with real-time processing",
            "video_id": video_id,
            "filename": filename,
            "detection_type": detection_type,
            "status": "processing",
            "thumbnail_url": f"/thumbnails/{os.path.basename(thumbnail_path)}" if thumbnail_path else None
        }
    
    except Exception as e:
        logger.error(f"Video upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get dashboard statistics
@router.get("/dashboard/statistics")
async def get_dashboard_statistics():
    """Get real-time dashboard statistics"""
    try:
        # Get current statistics
        current_stats = {
            "total_cameras": len(active_detections),
            "active_detections": sum(len(d) for d in active_detections.values()),
            "current_alerts": len(await get_incidents(recent=True)),
            "system_status": "Optimal",
            "detection_counts": {
                "people": sum(1 for d in active_detections.values() for det in d if det.get("class_name") == "person"),
                "vehicles": sum(1 for d in active_detections.values() for det in d if det.get("class_name") in ["car", "truck"]),
                "objects": sum(1 for d in active_detections.values() for det in d if det.get("type") == "object"),
                "faces": sum(1 for d in active_detections.values() for det in d if det.get("type") == "face")
            }
        }
        
        return current_stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include any additional routers or endpoints
# For example, face tracking, specific detection routes, etc.

@router.post("/cameras/{camera_id}/inference/start")
async def start_camera_inference(
    camera_id: int,
    detection_types: List[str] = Query(None, description="Types of detection to run (object, theft, loitering)"),
    stream_source: Optional[str] = Query(None, description="Optional override for camera stream source")
):
    """
    Start real-time inference with multiple models on a camera stream
    
    Args:
        camera_id (int): Camera ID to process
        detection_types (List[str]): Types of detection to enable
        stream_source (str, optional): Override stream source
        
    Returns:
        Dict: Stream status information
    """
    try:
        # Get camera if exists
        cameras = await get_cameras()
        camera = next((c for c in cameras if c["id"] == camera_id), None)
        
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Use provided stream source or fallback to camera settings
        video_path = stream_source or camera.get("stream_url") or camera.get("video_path")
        if not video_path:
            raise HTTPException(status_code=400, detail="No stream source provided and no URL configured for camera")
        
        # Import the detection service (to avoid circular imports)
        from app.services.detection_service import start_model_inference
        
        # Start inference
        result = await start_model_inference(
            camera_id=camera_id,
            stream_source=video_path,
            detection_types=detection_types
        )
        
        # Update camera status in database
        await update_camera_status(camera_id, "online")
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/{camera_id}/inference/stop")
async def stop_camera_inference(camera_id: int):
    """
    Stop real-time inference on a camera stream
    
    Args:
        camera_id (int): Camera ID to stop
        
    Returns:
        Dict: Status information
    """
    try:
        # Import the detection service
        from app.services.detection_service import stop_model_inference
        
        # Stop inference
        result = await stop_model_inference(camera_id)
        
        # Update camera status in database
        await update_camera_status(camera_id, "offline")
        
        return result
        
    except Exception as e:
        logger.error(f"Error stopping inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inference/active")
async def list_active_inferences():
    """
    Get all active inference processes
    
    Returns:
        Dict: Active inference information
    """
    try:
        # Import the detection service
        from app.services.detection_service import get_active_inferences
        
        # Get active inferences
        active = get_active_inferences()
        
        return {
            "status": "success",
            "active_count": len(active),
            "active_inferences": active
        }
        
    except Exception as e:
        logger.error(f"Error getting active inferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/inference")
async def inference_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time inference with multiple models
    
    Handles streaming video frames with detection overlays and real-time detection results.
    Clients can subscribe to specific cameras and detection types.
    """
    client_id = None
    subscribed_cameras = set()
    
    try:
        # Establish connection
        client_id = f"inference_{uuid.uuid4().hex[:8]}"
        await websocket_manager.connect(websocket, client_id)
        logger.info(f"Real-time inference WebSocket connected: {client_id}")
        
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id
        })
        
        # Handle incoming messages
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                action = message.get("action", "")
                
                if action == "subscribe":
                    # Subscribe to camera
                    camera_id = message.get("camera_id")
                    if not camera_id:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No camera_id provided"
                        })
                        continue
                    
                    # Add to subscribed cameras
                    subscribed_cameras.add(camera_id)
                    
                    # Import the detection service
                    from app.services.detection_service import active_streams
                    
                    # Check if camera is already streaming
                    if camera_id not in active_streams:
                        # Start inference if not already running
                        detection_types = message.get("detection_types", ["object", "theft", "loitering"])
                        
                        # Get camera info
                        cameras = await get_cameras()
                        camera = next((c for c in cameras if c["id"] == camera_id), None)
                        
                        if not camera:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Camera {camera_id} not found"
                            })
                            continue
                        
                        # Use camera stream URL
                        stream_url = camera.get("stream_url") or camera.get("video_path")
                        if not stream_url:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"No stream URL for camera {camera_id}"
                            })
                            continue
                        
                        # Start inference
                        from app.services.detection_service import start_model_inference
                        await start_model_inference(
                            camera_id=camera_id,
                            stream_source=stream_url,
                            detection_types=detection_types
                        )
                    
                    # Confirm subscription
                    await websocket.send_json({
                        "type": "subscribed",
                        "camera_id": camera_id
                    })
                
                elif action == "unsubscribe":
                    # Unsubscribe from camera
                    camera_id = message.get("camera_id")
                    if camera_id and camera_id in subscribed_cameras:
                        subscribed_cameras.remove(camera_id)
                        
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "camera_id": camera_id
                        })
                
                elif action == "list_cameras":
                    # Get available cameras
                    cameras = await get_cameras()
                    
                    # Import the detection service
                    from app.services.detection_service import active_streams
                    
                    # Add active status
                    for camera in cameras:
                        camera["is_active_inference"] = camera["id"] in active_streams
                    
                    await websocket.send_json({
                        "type": "camera_list",
                        "cameras": cameras
                    })
                
                elif action == "start_inference":
                    # Start inference on a camera
                    camera_id = message.get("camera_id")
                    detection_types = message.get("detection_types", ["object", "theft", "loitering"])
                    
                    if not camera_id:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No camera_id provided"
                        })
                        continue
                    
                    # Get camera info
                    cameras = await get_cameras()
                    camera = next((c for c in cameras if c["id"] == camera_id), None)
                    
                    if not camera:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Camera {camera_id} not found"
                        })
                        continue
                    
                    # Use camera stream URL
                    stream_url = camera.get("stream_url") or camera.get("video_path")
                    if not stream_url:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"No stream URL for camera {camera_id}"
                        })
                        continue
                    
                    # Start inference
                    from app.services.detection_service import start_model_inference
                    result = await start_model_inference(
                        camera_id=camera_id,
                        stream_source=stream_url,
                        detection_types=detection_types
                    )
                    
                    # Subscribe to the camera
                    subscribed_cameras.add(camera_id)
                    
                    # Return result
                    await websocket.send_json({
                        "type": "inference_started",
                        "camera_id": camera_id,
                        "result": result
                    })
                
                elif action == "stop_inference":
                    # Stop inference on a camera
                    camera_id = message.get("camera_id")
                    
                    if not camera_id:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No camera_id provided"
                        })
                        continue
                    
                    # Stop inference
                    from app.services.detection_service import stop_model_inference
                    result = await stop_model_inference(camera_id)
                    
                    # Unsubscribe from the camera
                    if camera_id in subscribed_cameras:
                        subscribed_cameras.remove(camera_id)
                    
                    # Return result
                    await websocket.send_json({
                        "type": "inference_stopped",
                        "camera_id": camera_id,
                        "result": result
                    })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown action: {action}"
                    })
            
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from inference client {client_id}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            
            except Exception as e:
                logger.error(f"Error in inference WebSocket: {str(e)}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except:
                    pass
    
    except WebSocketDisconnect:
        logger.info(f"Inference WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Unexpected error in inference WebSocket: {str(e)}")
    finally:
        # Clean up
        if client_id:
            websocket_manager.disconnect(websocket, client_id)
        logger.info(f"Inference WebSocket connection closed: {client_id}")