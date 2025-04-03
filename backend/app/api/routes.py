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

# Helper functions for incident creation
async def save_incident_frame(frame, incident_type, camera_id):
    """Save frame as an incident image"""
    try:
        # Create incidents directory if it doesn't exist
        incidents_dir = os.path.join("incidents")
        os.makedirs(incidents_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{incident_type}_{camera_id}_{timestamp}.jpg"
        filepath = os.path.join(incidents_dir, filename)
        
        # Save the frame
        cv2.imwrite(filepath, frame)
        logger.info(f"Saved incident frame to {filepath}")
        
        return filepath
    except Exception as e:
        logger.error(f"Error saving incident frame: {str(e)}")
        return None

async def create_incident(camera_id, incident_type, confidence, frame_path, metadata=None):
    """Create an incident record in the database"""
    try:
        # In a real application, this would be stored in a database
        # For now, we'll just log it
        logger.info(f"Created incident: Camera {camera_id}, Type: {incident_type}, Confidence: {confidence:.2f}")
        logger.info(f"  Frame path: {frame_path}")
        if metadata:
            logger.info(f"  Metadata: {metadata}")
        
        # Broadcast incident to all connected clients
        await websocket_manager.broadcast({
            "type": "new_incident",
            "camera_id": camera_id,
            "incident_type": incident_type,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "frame_path": frame_path
        })
        
        return True
    except Exception as e:
        logger.error(f"Error creating incident: {str(e)}")
        return False

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
            "is_video_file": use_video
        })
        
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps = 0
        critical_detection_count = 0
        
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
            
            # Process theft detection if enabled
            if theft_model and "theft" in detection_types:
                theft_results = theft_model.detect(frame.copy())
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
            
            # Process loitering detection if enabled
            if loitering_model and "loitering" in detection_types:
                loitering_results = loitering_model.detect(frame.copy(), camera_id=camera_id)
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
            
            # Encode frame for transmission
            success, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
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
            await asyncio.sleep(0.05)  # Adjust for desired frame rate (0.05 = ~20 FPS max)
    
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