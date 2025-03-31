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

@router.get("/videos/{video_id}/detections")
async def get_video_detections(video_id: int):
    """Get all detections for a specific video"""
    return await get_detections(video_id)

@router.post("/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detection_type: str = Query(..., description="Type of detection: theft, loitering, face_detection")
):
    """Upload video for processing with real-time detection updates"""
    if detection_type not in ["theft", "loitering", "face_detection"]:
        raise HTTPException(status_code=400, detail="Invalid detection type")
    
    try:
        # Generate unique video ID and paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Set up file paths
        raw_path = settings.UPLOAD_DIR / f"{unique_id}_{file.filename}"
        processed_path = settings.PROCESSED_DIR / f"{unique_id}_processed_{file.filename}"
        thumbnail_path = settings.THUMBNAILS_DIR / f"{unique_id}_thumb.jpg"
        
        # Save uploaded file
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add video to database
        video_data = {
            "name": file.filename,
            "file_path": str(raw_path),
            "processed_file_path": str(processed_path),
            "thumbnail_path": str(thumbnail_path),
            "status": "processing",
            "detection_type": detection_type,
            "upload_time": datetime.now()
        }
        video_id = await add_video(video_data)
        
        # Start processing in background
        background_tasks.add_task(
            process_video_background,
            str(raw_path),
            video_id,
            detection_type
        )
        
        return {
            "message": "Video uploaded and processing started",
            "video_id": video_id,
            "filename": file.filename,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos/{video_id}/status")
async def get_video_status(video_id: int):
    """Get the processing status of a video"""
    videos = await get_all_videos()
    video = next((v for v in videos if v["id"] == video_id), None)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return video

@router.get("/videos/{video_id}/stream")
async def stream_video(video_id: int):
    """Stream processed video with real-time detections"""
    videos = await get_all_videos()
    video = next((v for v in videos if v["id"] == video_id), None)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not video["processed_file_path"]:
        raise HTTPException(status_code=400, detail="Video not processed yet")
    
    async def generate_frames():
        cap = cv2.VideoCapture(video["processed_file_path"])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Convert to base64 for streaming
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
            
            # Send frame through WebSocket
            await manager.broadcast(json.dumps({
                "type": "frame",
                "video_id": video_id,
                "frame": frame_base64
            }))
            
            # Control frame rate
            await asyncio.sleep(1/30)  # 30 FPS
        
        cap.release()
    
    # Start frame generation in background
    asyncio.create_task(generate_frames())
    
    return {"message": "Streaming started"}

@router.post("/videos/process-all")
async def process_all_videos(background_tasks: BackgroundTasks):
    """Process all videos in the raw folder with detections"""
    try:
        # Define video paths
        videos = [
            {
                "id": 1,
                "name": "Cleaning Section",
                "path": r"E:\code\EMB Global\backend\uploads\raw\Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4"
            }
        ]

        processed_videos = []
        
        for video in videos:
            if not Path(video["path"]).exists():
                logger.warning(f"Video not found: {video['path']}")
                continue

            # Create output path for processed video
            output_filename = f"processed_{Path(video['path']).name}"
            output_path = settings.PROCESSED_DIR / output_filename
            
            # Add video to database
            video_data = {
                "name": video["name"],
                "file_path": video["path"],
                "processed_file_path": str(output_path),
                "status": "processing",
                "detection_type": "all",
                "upload_time": datetime.now()
            }
            
            video_id = await add_video(video_data)
            processed_videos.append({
                "video_id": video_id,
                "name": video["name"],
                "input_path": video["path"],
                "output_path": str(output_path)
            })
            
            # Start processing in background
            background_tasks.add_task(
                process_video_background,
                video["path"],
                video_id,
                "all"
            )

        return {
            "message": "Started processing all videos",
            "videos": processed_videos
        }
        
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cameras")
async def get_cameras():
    """Get available cameras including raw video feeds"""
    try:
        # Define absolute paths for the videos
        video_paths = [
            
            r"E:/code/EMB Global/backend/uploads/raw/Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4"
        ]
        
        # Create camera entries for the videos
        cameras = []
        for idx, video_path in enumerate(video_paths, 1):
            if Path(video_path).exists():
                name = "Cleaning Section" if idx == 3 else f"Camera Feed {idx}"
                cameras.append({
                    "id": idx,
                    "name": name,
                    "status": "online",
                    "video_path": video_path,
                    "stream_url": f"/api/cameras/{idx}/live",
                    "fps": 30
                })
                logger.info(f"Added camera {idx} with video path: {video_path}")
            else:
                logger.warning(f"Video file not found: {video_path}")
        
        if not cameras:
            logger.warning("No video files found")
            raise HTTPException(status_code=404, detail="No cameras available")
        
        return cameras
        
    except Exception as e:
        logger.error(f"Error getting cameras: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cameras/{camera_id}")
async def get_camera(camera_id: int):
    """Get specific camera details"""
    cameras = await get_cameras()
    camera = next((c for c in cameras if c["id"] == camera_id), None)
    
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return camera

@router.get("/cameras/{camera_id}/live")
async def get_live_camera_feed(camera_id: int):
    """Stream video feed with real-time processing"""
    try:
        cameras = await get_cameras()
        camera = next((c for c in cameras if c["id"] == camera_id), None)
        
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        video_path = camera.get("video_path")
        if not video_path or not Path(video_path).exists():
            raise HTTPException(status_code=400, detail=f"Video file not found at path: {video_path}")
        
        logger.info(f"Opening video file: {video_path}")

        async def generate_frames():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail=f"Failed to open video feed at path: {video_path}")
            
            logger.info(f"Successfully opened video feed for camera {camera_id}")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start
                        continue

                    try:
                        # Process frame for detections
                        detections = await video_processor.process_frame(frame)
                        
                        # Draw detections on frame
                        annotated_frame = frame.copy()
                        for det in detections:
                            x1, y1, x2, y2 = map(int, det["bbox"])
                            confidence = det["confidence"]
                            class_name = det["class_name"]
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label with confidence
                            label = f"{class_name}: {confidence:.2f}"
                            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            y1 = max(y1, label_size[1])
                            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - baseline),
                                        (x1 + label_size[0], y1), (0, 255, 0), cv2.FILLED)
                            cv2.putText(annotated_frame, label, (x1, y1 - baseline),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        # Convert frame to base64 for streaming
                        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send frame and detections through WebSocket
                        message = {
                            "type": "live_detection",
                            "camera_id": camera_id,
                            "frame": frame_base64,
                            "detections": [
                                {
                                    "type": det["type"],
                                    "class_name": det["class_name"],
                                    "confidence": float(det["confidence"]),
                                    "bbox": det["bbox"]
                                }
                                for det in detections
                            ],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        await manager.broadcast(json.dumps(message))
                        
                        # Store detections in database if there are any
                        if detections:
                            await store_detections(camera_id, detections)
                            
                            # Save annotated frame
                            frame_path = settings.FRAMES_DIR / f"camera_{camera_id}_frame_{int(time.time())}.jpg"
                            cv2.imwrite(str(frame_path), annotated_frame)

                    except Exception as e:
                        logger.error(f"Error processing frame: {str(e)}")
                        continue

                    await asyncio.sleep(1/30)  # Maintain 30 FPS
                    
            except Exception as e:
                logger.error(f"Error in frame generation: {str(e)}")
                raise
            finally:
                cap.release()

        # Start frame generation
        background_tasks = BackgroundTasks()
        background_tasks.add_task(generate_frames)
        
        return {"status": "success", "message": "Live stream started", "camera_id": camera_id}
        
    except Exception as e:
        logger.error(f"Error in live camera feed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts(filter: str = "recent"):
    """Get all alerts"""
    current_time = datetime.now()
    alerts = [
        {
            "id": 1,
            "timestamp": current_time.isoformat(),
            "location": "Front Entrance",
            "type": "Unauthorized Access",
            "description": "Unauthorized person detected",
            "image": "https://example.com/image1.jpg",
            "video_url": "https://example.com/video1.mp4",
            "severity": "high"
        }
    ]
    return alerts

@router.get("/alerts/mall-structure")
async def get_mall_structure():
    """Get mall structure data"""
    return [
        {
            "id": 1,
            "name": "Entrance A",
            "crowd_density": 85,
            "bounds": [[0, 0], [20, 20]],
            "fill_color": "#FF4D4D"
        }
    ]

@router.get("/billing-activity")
async def get_billing_activity(filter: str = "all"):
    """Get billing activity data"""
    return [
        {
            "id": 1,
            "transaction_id": "TXN001",
            "customer_id": "CUST001",
            "timestamp": datetime.now().isoformat(),
            "products": [
                {"name": "Product 1", "quantity": 2, "price": 100}
            ],
            "total_amount": 200,
            "status": "completed",
            "suspicious": False
        }
    ]

@router.get("/customer-data")
async def get_customer_data(
    gender: Optional[str] = None,
    date: Optional[str] = None,
    time_period: Optional[str] = None
):
    """Get customer data with optional filters"""
    return [
        {
            "id": 1,
            "image_url": "https://example.com/customer1.jpg",
            "gender": "Male",
            "entry_time": "10:00 AM",
            "entry_date": "2024-03-20",
            "age_group": "25-34",
            "clothing_color": "Blue",
            "notes": "Regular customer"
        }
    ]

@router.get("/daily-report")
async def get_daily_report(date: str):
    """Get daily report data"""
    return {
        "total_entries": 500,
        "total_purchases": 200,
        "no_purchase": 300,
        "peak_hour": "12 PM - 1 PM",
        "average_time_spent": "25 minutes",
        "hourly_breakdown": [
            {"hour": "6 AM", "entries": 20, "purchases": 10},
            {"hour": "7 AM", "entries": 30, "purchases": 15}
        ]
    }

@router.get("/system-status")
async def get_system_status():
    """Get system status information"""
    return {
        "cameras": [
            {"id": 1, "name": "Camera A", "status": "Online", "fps": 30}
        ],
        "model_performance": {
            "is_working": True,
            "accuracy": "98.5%",
            "true_positives": 1200,
            "false_positives": 15
        },
        "frame_skipping": [
            {"id": 1, "camera_id": 1, "skipped_frames": 5}
        ]
    }

@router.get("/dashboard/statistics")
async def get_dashboard_statistics():
    """Get real-time dashboard statistics"""
    try:
        current_stats = {
            "total_cameras": len(active_detections),
            "active_detections": sum(len(d) for d in active_detections.values()),
            "current_alerts": len(await get_incidents(recent=True)),
            "system_status": "Optimal",
            "detection_counts": {
                "people": sum(1 for d in active_detections.values() for det in d if det["class_name"] == "person"),
                "vehicles": sum(1 for d in active_detections.values() for det in d if det["class_name"] in ["car", "truck"]),
                "objects": sum(1 for d in active_detections.values() for det in d if det["type"] == "object"),
                "faces": sum(1 for d in active_detections.values() for det in d if det["type"] == "face")
            }
        }
        
        return current_stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/incidents")
async def get_dashboard_incidents():
    """Get dashboard incidents"""
    return [
        {
            "id": 1,
            "title": "Unauthorized Access",
            "location": "Front Entrance",
            "time": "14:35",
            "severity": "high"
        }
    ]

@router.get("/dashboard/crowd-density")
async def get_crowd_density():
    """Get crowd density data"""
    return [
        {
            "id": 1,
            "name": "Entrance A",
            "density": 85,
            "status": "High"
        }
    ]

@router.get("/detections/current")
async def get_current_detections():
    """Get current detections for all cameras"""
    try:
        detections = await get_latest_detections()
        return {
            "status": "success",
            "detections": detections
        }
    except Exception as e:
        logger.error(f"Error getting current detections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    """Detect loitering in video"""
    try:
        # Create output path
        output_path = settings.PROCESSED_DIR / f"loitering_{Path(video_path).name}"
        
        # Start loitering detection in background
        background_tasks.add_task(
            video_processor.process_loitering_detection,
            video_path,
            str(output_path),
            threshold_time
        )
        
        return {
            "message": "Loitering detection started",
            "output_path": str(output_path)
        }
    except Exception as e:
        logger.error(f"Error starting loitering detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/videos/theft-detection")
async def detect_theft(
    background_tasks: BackgroundTasks,
    video_path: str = Query(..., description="Path to input video"),
    hand_stay_time: int = Query(2, description="Time threshold for suspicious hand positions")
):
    """Detect suspicious behavior in video"""
    try:
        # Create output paths
        output_path = settings.PROCESSED_DIR / f"theft_{Path(video_path).name}"
        screenshot_dir = settings.FRAMES_DIR / f"theft_{int(time.time())}"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Start theft detection in background
        background_tasks.add_task(
            video_processor.process_theft_detection,
            video_path,
            str(output_path),
            str(screenshot_dir),
            hand_stay_time
        )
        
        return {
            "message": "Theft detection started",
            "output_path": str(output_path),
            "screenshot_dir": str(screenshot_dir)
        }
    except Exception as e:
        logger.error(f"Error starting theft detection: {str(e)}")
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