from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, WebSocket, WebSocketDisconnect
import requests
import tempfile
import pickle
import uuid
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import base64
import numpy as np
import asyncio
import time

from ..core.config import settings
from ..core.websocket import websocket_manager
from ..services.video_processor import video_processor, VideoProcessor

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
        # Look up job status
        if job_id not in processing_jobs:
            raise HTTPException(status_code=404, detail=f"Job ID not found: {job_id}")
            
        job_info = processing_jobs[job_id]
        
        return {
            "job_id": job_id,
            "status": job_info.get("status", "unknown"),
            "progress": job_info.get("progress", 0),
            "output_path": job_info.get("output_path"),
            "start_time": job_info.get("start_time"),
            "end_time": job_info.get("end_time"),
            "error": job_info.get("error")
        }
        
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Video Processor Configuration Endpoint
@router.post("/config/display-mode")
async def configure_display_mode(
    mode: str = Query("raw", description="Display mode: 'raw' (no visualization) or 'detection' (with visualization)"),
    run_detection: bool = Query(True, description="Whether to run detection in background")
):
    """
    Configure how video streams are displayed and processed
    
    Args:
        mode: Display mode - 'raw' for clean video, 'detection' for visualization
        run_detection: Whether to run detection processes in background
        
    Returns:
        Dict: Updated configuration settings
    """
    try:
        # Configure the video processor
        result = video_processor.configure_display_mode(mode=mode, run_detection=run_detection)
        
        return {
            "message": f"Display mode configured to '{mode}', background detection: {run_detection}",
            "config": result
        }
        
    except Exception as e:
        logger.error(f"Error configuring display mode: {str(e)}")
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

# Utility function for frame conversion
async def convert_frame_to_bytes(frame, quality=70, max_width=640):
    """Optimize frame for streaming by resizing and compressing"""
    try:
        # Resize if needed for bandwidth optimization
        h, w = frame.shape[:2]
        if w > max_width:
            ratio = max_width / w
            new_h = int(h * ratio)
            frame = cv2.resize(frame, (max_width, new_h))
            
        # Compress to JPEG format
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        # Convert to base64 for easy transmission
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        return encoded_frame
    except Exception as e:
        logger.error(f"Frame conversion error: {str(e)}")
        return None

# Utility function for saving images optimized for storage
def optimize_frame_for_storage(frame, quality=85, max_width=1280):
    """Optimize frame for storage by resizing and compression"""
    # Resize if needed
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_h, new_w = int(h * scale), max_width
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Compress
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    
    return buffer

async def save_incident_frame(frame, detection_type, camera_id):
    """
    Save incident frame to disk
    
    Args:
        frame: Frame to save
        detection_type: Type of detection (theft, loitering)
        camera_id: Camera identifier
        
    Returns:
        str: Path to saved image
    """
    try:
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Create directory if not exists
        incident_dir = os.path.join(settings.INCIDENTS_DIR, detection_type)
        os.makedirs(incident_dir, exist_ok=True)
        
        # Create filename
        filename = f"{detection_type}_{camera_id}_{timestamp}.jpg"
        filepath = os.path.join(incident_dir, filename)
        
        # Optimize and save frame
        buffer = optimize_frame_for_storage(frame)
        with open(filepath, 'wb') as f:
            f.write(buffer)
        
        return filepath
    except Exception as e:
        logger.error(f"Error saving incident frame: {str(e)}")
        return None

async def create_incident(camera_id, incident_type, confidence, frame_path, metadata=None):
    """
    Create incident record in database
    
    Args:
        camera_id: Camera identifier
        incident_type: Type of incident (theft, loitering)
        confidence: Detection confidence
        frame_path: Path to saved frame
        metadata: Additional metadata about the incident
        
    Returns:
        str: Incident ID
    """
    try:
        # Prepare incident data
        incident_data = {
            "type": incident_type,
            "timestamp": datetime.now(),
            "location": f"Camera {camera_id}",
            "description": f"{incident_type.capitalize()} detected with {confidence:.2f} confidence",
            "image_path": frame_path,
            "video_url": None,  # For live feed, no video URL
            "severity": "high" if confidence > 0.8 else "medium",
            "confidence": confidence,
            "is_resolved": False,
            "metadata": metadata or {}
        }
        
        # Add to database
        # Note: In a real implementation, this would use your database schema
        # For now, we'll just log it
        logger.info(f"Created incident: {incident_data}")
        incident_id = str(uuid.uuid4())
        
        return incident_id
    except Exception as e:
        logger.error(f"Error creating incident: {str(e)}")
        return None

# Add a new endpoint to fetch recent detections
@router.get("/detections/recent")
async def get_recent_detections(
    limit: int = Query(10, description="Maximum number of detections to retrieve"),
    detection_type: Optional[str] = Query(None, description="Filter by detection type (theft, loitering)")
):
    """
    Get recent detections for fallback when WebSocket is disconnected
    """
    try:
        # Import here to avoid circular imports
        from database import async_session, Detection, select
        from sqlalchemy import desc
        
        async with async_session() as session:
            query = select(Detection).order_by(desc(Detection.timestamp))
            
            if detection_type:
                query = query.where(Detection.detection_type == detection_type)
                
            query = query.limit(limit)
            result = await session.execute(query)
            detections = result.scalars().all()
            
            # Convert to serializable format
            detection_list = []
            for detection in detections:
                detection_dict = {
                    "id": detection.id,
                    "timestamp": detection.timestamp.isoformat(),
                    "frame_number": detection.frame_number,
                    "detection_type": detection.detection_type,
                    "confidence": detection.confidence,
                    "bbox": detection.bbox,
                    "class_name": detection.class_name,
                    "image_path": detection.image_path,
                    "camera_id": detection.camera_id,
                }
                if detection.detection_metadata:
                    detection_dict["metadata"] = detection.detection_metadata
                
                detection_list.append(detection_dict)
            
            return detection_list
            
    except Exception as e:
        logger.error(f"Error retrieving recent detections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhance realtime_detection WebSocket for better frontend integration
@router.websocket("/ws/realtime-detection")
async def realtime_detection_websocket(
    websocket: WebSocket,
    detection_type: str = "both",
    camera_id: str = "0",
    use_mock: bool = False,
    use_video: bool = False
):
    """
    WebSocket endpoint for real-time inferencing
    Sends camera frames and detection results to the client
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected - Camera ID: {camera_id}, Detection Type: {detection_type}")
    
    # Clean detection_type parameter
    detection_types = []
    if detection_type == "both":
        detection_types = ["theft", "loitering"]
    elif detection_type in ["theft", "loitering"]:
        detection_types = [detection_type]
    
    # Use the models from the video_processor instance
    theft_model = None
    loitering_model = None
    if "theft" in detection_types:
        theft_model = video_processor.theft_detection_model
    if "loitering" in detection_types:
        loitering_model = video_processor.loitering_detection_model
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection_established", 
            "camera_id": camera_id,
            "detection_types": detection_types,
            "is_mock": use_mock,
            "is_video_file": use_video,
            "display_mode": video_processor.display_mode,
            "background_detection": video_processor.run_detection_in_background
        })
        
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps = 0
        critical_detection_count = 0
        
        # Throttle variables for incident recording
        last_incident_time = {
            "theft": time.time() - 10,  # Initialize to allow immediate first incident
            "loitering": time.time() - 10
        }
        incident_throttle_seconds = 5.0  # Only record an incident every 5 seconds per type
        
        while True:
            # Get frame from camera or mock data
            frame, is_mock, video_info, is_video_file = await video_processor.get_video_frame(
                camera_id=camera_id, 
                use_mock=use_mock,
                use_video=use_video
            )
            
            if frame is None:
                # If no frame, send error message
                await websocket.send_json({
                    "type": "error",
                    "message": "No video frame available"
                })
                # Wait before retrying
                await asyncio.sleep(1)
                continue
            
            # Update FPS calculation
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            if current_time - last_fps_update >= 1.0:  # Update FPS every second
                fps = frame_count / (current_time - start_time)
                last_fps_update = current_time
            
            # Object detection results
            detections = {}
            has_critical_detection = False
            
            # Skip detection if it's disabled completely
            if video_processor.display_mode == "raw" and not video_processor.run_detection_in_background:
                # Skip all detection when raw mode and background detection disabled
                pass
            else:
                # Process theft detection if enabled
                if theft_model and "theft" in detection_types:
                    # Use a copy of the frame for detection if we're not showing visualizations
                    detection_frame = frame.copy() if video_processor.display_mode == "raw" else frame
                    
                    theft_results = theft_model.detect(
                        frame=detection_frame, 
                        camera_id=camera_id,
                        draw_visualization=(video_processor.display_mode == "detection")
                    )
                    detections["theft"] = theft_results
                    
                    # Check if this is a critical detection (high confidence)
                    if theft_results.get("detected", False) and theft_results.get("confidence", 0) > 0.8:
                        has_critical_detection = True
                        critical_detection_count += 1
                        
                        # Log critical detections
                        logger.warning(f"Critical theft detection - Camera {camera_id}: "
                                      f"Confidence: {theft_results.get('confidence', 0):.2f}")
                        
                        # Create incident record for high-confidence detections
                        if theft_results.get("confidence", 0) > 0.85:
                            # Check if enough time has passed since last incident
                            current_time = time.time()
                            if current_time - last_incident_time["theft"] >= incident_throttle_seconds:
                                # Save frame for incident
                                incident_frame_path = await save_incident_frame(frame, "theft", camera_id)
                                
                                # Create incident in database
                                await create_incident(
                                    camera_id=camera_id,
                                    incident_type="theft",
                                    confidence=theft_results.get("confidence", 0),
                                    frame_path=incident_frame_path,
                                    metadata={
                                        "bounding_boxes": theft_results.get("bounding_boxes", []),
                                        "is_mock": is_mock,
                                        "is_video_file": is_video_file
                                    }
                                )
                                
                                # Update last incident time
                                last_incident_time["theft"] = current_time
                
                # Process loitering detection if enabled
                if loitering_model and "loitering" in detection_types:
                    # Use a copy of the frame for detection if we're not showing visualizations
                    detection_frame = frame.copy() if video_processor.display_mode == "raw" else frame
                    
                    loitering_results = loitering_model.detect(
                        frame=detection_frame,
                        camera_id=camera_id,
                        draw_visualization=(video_processor.display_mode == "detection")
                    )
                    detections["loitering"] = loitering_results
                    
                    # Check for critical loitering detection (long duration)
                    if loitering_results.get("detected", False) and loitering_results.get("duration", 0) > 12.0:
                        has_critical_detection = True
                        critical_detection_count += 1
                        
                        # Log critical detections
                        logger.warning(f"Critical loitering detection - Camera {camera_id}: "
                                     f"Duration: {loitering_results.get('duration', 0):.2f}s")
                        
                        # Create incident record for long-duration loitering
                        if loitering_results.get("duration", 0) > 15.0:
                            # Check if enough time has passed since last incident
                            current_time = time.time()
                            if current_time - last_incident_time["loitering"] >= incident_throttle_seconds:
                                # Save frame for incident
                                incident_frame_path = await save_incident_frame(frame, "loitering", camera_id)
                                
                                # Create incident in database
                                await create_incident(
                                    camera_id=camera_id,
                                    incident_type="loitering",
                                    confidence=min(1.0, loitering_results.get("duration", 0) / 20.0),  # Convert duration to confidence
                                    frame_path=incident_frame_path,
                                    metadata={
                                        "duration": loitering_results.get("duration", 0),
                                        "regions": loitering_results.get("regions", []),
                                        "is_mock": is_mock,
                                        "is_video_file": is_video_file
                                    }
                                )
                                
                                # Update last incident time
                                last_incident_time["loitering"] = current_time
            
            # Prepare frame for transmission
            h, w = frame.shape[:2]
            # Resize frame to 640p width for faster transmission
            if w > 640:
                scale_factor = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale_factor)), interpolation=cv2.INTER_AREA)
            
            # Encode frame for transmission
            success, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            if not success:
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to encode frame"
                })
                await asyncio.sleep(0.1)
                continue
                
            # Convert to base64 for transmission
            frame_base64 = base64.b64encode(encoded_frame).decode('utf-8')
            
            # Compose message with frame and detections
            message = {
                "type": "inference_result",
                "frame": frame_base64,
                "detections": detections,
                "ui_metadata": {
                    "fps": round(fps, 1),
                    "frame_count": frame_count,
                    "elapsed_time": round(elapsed, 2),
                    "camera_id": camera_id,
                    "has_critical_detection": has_critical_detection,
                    "critical_detection_count": critical_detection_count,
                    "is_mock": is_mock,
                    "is_video_file": is_video_file
                }
            }
            
            # Add video file info if available
            if is_video_file and video_info:
                message["ui_metadata"]["video_info"] = video_info
            
            # Send message to client
            await websocket.send_json(message)
            
            # Throttle to avoid overwhelming the client
            await asyncio.sleep(0.02)  # Adjusted for higher frame rate (0.02 = ~50 FPS max)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: Camera ID {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        # Try to send error message to client
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass
    finally:
        # Cleanup resources
        video_processor.release_video_cameras(camera_id)
        logger.info(f"Cleaned up resources for camera ID {camera_id}")

@router.websocket("/ws/loitering-stream")
async def loitering_stream(websocket: WebSocket, camera_id: str = None):
    """Legacy endpoint redirecting to the unified detection endpoint"""
    await realtime_detection_websocket(websocket, detection_type="loitering", camera_id=camera_id)

@router.websocket("/ws/theft-stream")
async def theft_stream(websocket: WebSocket, camera_id: str = None):
    """Legacy endpoint redirecting to the unified detection endpoint"""
    await realtime_detection_websocket(websocket, detection_type="theft", camera_id=camera_id)