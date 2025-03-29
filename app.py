from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Query, Path, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional, Any, Set
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import cv2
import numpy as np
import time
import uvicorn
import torch
from datetime import datetime, timedelta
import uuid
import json
from pydantic import BaseModel
import asyncio
import shutil
from pathlib import Path
import threading
import queue
import random
import logging
import sys
from typing import Union
import base64
from io import BytesIO
from PIL import Image
from backend.database import (
    init_db,
    get_all_videos,
    get_detections,
    get_incidents,
    add_video,
    add_detection,
    add_incident,
    update_video_status,
    add_customer_data
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Security Dashboard API", 
              description="API for security monitoring, alerts and analytics",
              version="1.0.0")

# CORS settings
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create folders for storing data
try:
    UPLOAD_DIR = Path("uploads")
    PROCESSED_DIR = Path("processed")
    FRAMES_DIR = Path("frames")
    ALERTS_DIR = Path("alerts")
    THUMBNAILS_DIR = Path("thumbnails")
    MODELS_DIR = Path("models")

    for dir_path in [UPLOAD_DIR, PROCESSED_DIR, FRAMES_DIR, ALERTS_DIR, THUMBNAILS_DIR, MODELS_DIR]:
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
except Exception as e:
    logger.error(f"Error creating directories: {str(e)}")
    raise

# Model paths
FACE_MODEL_PATH = Path("E:/code/EMB Global/models/yolov8n-face.pt")
POSE_MODEL_PATH = Path("E:/code/EMB Global/models/yolov8n-pose.pt")
OBJECT_MODEL_PATH = Path("E:/code/EMB Global/models/yolov5s.pt")

# Add new model paths
FACE_EXTRACTION_MODEL_PATH = Path("E:/code/EMB Global/models/yolov8n-face.pt")
LOITERING_MODEL_PATH = Path("E:/code/EMB Global/models/yolov5s.pt")
THEFT_MODEL_PATH = Path("E:/code/EMB Global/models/yolo11n-pose.pt")

# Load models if YOLO is available
try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    import torch.serialization
    from torch.nn.modules.container import Sequential
    from torch.nn.modules import Conv2d, BatchNorm2d, SiLU, Module, ModuleList
    YOLO_AVAILABLE = True
    logger.info("YOLO package is available")
    
    # Add all required safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([
        DetectionModel,
        Sequential,
        Conv2d,
        BatchNorm2d,
        SiLU,
        Module,
        ModuleList
    ])
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO package is not available")

face_model = None
pose_model = None
object_model = None

# Load additional models
face_extraction_model = None
loitering_model = None
theft_model = None

if YOLO_AVAILABLE:
    try:
        # First try to load the object model as it's most critical
        if OBJECT_MODEL_PATH.exists():
            logger.info(f"Loading object detection model from {OBJECT_MODEL_PATH}")
            object_model = YOLO(str(OBJECT_MODEL_PATH), task='detect')
            logger.info("Object detection model loaded successfully")
        else:
            logger.error(f"Object model not found at {OBJECT_MODEL_PATH}")
            raise FileNotFoundError(f"Object model not found at {OBJECT_MODEL_PATH}")

        # Then load other models
        if FACE_MODEL_PATH.exists():
            face_model = YOLO(str(FACE_MODEL_PATH), task='detect')
            logger.info("Face detection model loaded successfully")
        else:
            logger.warning(f"Face model not found at {FACE_MODEL_PATH}")

        if POSE_MODEL_PATH.exists():
            pose_model = YOLO(str(POSE_MODEL_PATH), task='detect')
            logger.info("Pose detection model loaded successfully")
        else:
            logger.warning(f"Pose model not found at {POSE_MODEL_PATH}")

        # Load face extraction model
        if FACE_EXTRACTION_MODEL_PATH.exists():
            face_extraction_model = YOLO(str(FACE_EXTRACTION_MODEL_PATH))
            logger.info("Face extraction model loaded successfully")
        else:
            logger.warning(f"Face extraction model not found at {FACE_EXTRACTION_MODEL_PATH}")

        # Load loitering model
        if LOITERING_MODEL_PATH.exists():
            loitering_model = YOLO(str(LOITERING_MODEL_PATH))
            logger.info("Loitering detection model loaded successfully")
        else:
            logger.warning(f"Loitering model not found at {LOITERING_MODEL_PATH}")

        # Load theft model
        if THEFT_MODEL_PATH.exists():
            theft_model = YOLO(str(THEFT_MODEL_PATH))
            logger.info("Theft detection model loaded successfully")
        else:
            logger.warning(f"Theft model not found at {THEFT_MODEL_PATH}")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Verify model initialization
if not object_model:
    logger.error("Object detection model failed to initialize")
    raise RuntimeError("Object detection model failed to initialize")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

# Pydantic models for request/response validation
class VideoInfo(BaseModel):
    id: int
    name: str
    status: str
    upload_time: str
    detection_type: str
    processed_file_path: Optional[str]
    thumbnail_path: Optional[str]

class DetectionInfo(BaseModel):
    id: int
    timestamp: str
    frame_number: int
    detection_type: str
    confidence: float
    bbox: List[float]
    class_name: Optional[str]
    image_path: str
    detection_metadata: Optional[Dict]

class IncidentInfo(BaseModel):
    id: int
    timestamp: str
    location: str
    type: str
    description: str
    image: str
    video_url: str
    severity: str

class CameraStatus(BaseModel):
    id: int
    name: str
    status: str
    stream_url: str

class Incident(BaseModel):
    id: int
    timestamp: str
    location: str
    type: str
    description: str
    image: str
    video_url: str
    severity: str

class BillingActivity(BaseModel):
    id: int
    transaction_id: str
    customer_id: str
    timestamp: str
    products: List[dict]
    total_amount: float
    status: str
    suspicious: bool

class CustomerData(BaseModel):
    id: int
    image_url: str
    gender: str
    entry_time: str
    entry_date: str
    age_group: str
    clothing_color: str
    notes: Optional[str]

# Video processing queue
video_queue = queue.Queue()
processing_threads = []

def process_video(video_path: str, video_id: int, detection_type: str):
    """Process video and store detections"""
    try:
        # Update video status to processing
        asyncio.run(update_video_status(video_id, "processing"))
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        output_path = PROCESSED_DIR / f"processed_{video_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Select appropriate model based on detection type
        if detection_type == "theft":
            model = object_model
            target_classes = ["person", "backpack", "handbag", "suitcase"]
        elif detection_type == "loitering":
            model = pose_model
        else:  # face_detection
            model = face_model
        
        frame_number = 0
        detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for performance
            if frame_number % 5 == 0 and model:
                # Run detection
                results = model(frame)
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, 'boxes') else []
                    confidences = result.boxes.conf.cpu().numpy() if hasattr(result, 'boxes') else []
                    class_ids = result.boxes.cls.cpu().numpy() if hasattr(result, 'boxes') else []
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Filter detections based on type
                        if detection_type == "theft":
                            class_name = model.names[int(cls_id)]
                            if class_name not in target_classes:
                                continue
                        
                        # Save detection frame
                        frame_path = FRAMES_DIR / f"frame_{video_id}_{frame_number}_{i}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        
                        # Store detection data
                        detection_data = {
                            "video_id": video_id,
                            "timestamp": datetime.now(),
                            "frame_number": frame_number,
                            "detection_type": detection_type,
                            "confidence": float(conf),
                            "bbox": box.tolist(),
                            "class_name": model.names[int(cls_id)] if hasattr(model, 'names') else None,
                            "image_path": str(frame_path),
                            "detection_metadata": {
                                "fps": fps,
                                "frame_count": frame_count,
                                "resolution": f"{width}x{height}"
                            }
                        }
                        detections.append(detection_data)
                        
                        # Draw detection on frame
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{detection_type} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write processed frame
            out.write(frame)
            frame_number += 1
        
        cap.release()
        out.release()
        
        # Generate thumbnail
        thumbnail_path = THUMBNAILS_DIR / f"thumb_{video_id}.jpg"
        cv2.imwrite(str(thumbnail_path), frame)
        
        # Store detections in database
        for detection in detections:
            asyncio.run(add_detection(detection))
        
        # Update video status
        asyncio.run(update_video_status(video_id, "completed", str(output_path)))
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        asyncio.run(update_video_status(video_id, "failed"))

def video_processor_worker():
    """Worker function to process videos from the queue"""
    while True:
        try:
            task = video_queue.get(timeout=1)
            if task is None:  # Poison pill to stop the thread
                break
                
            video_path, video_id, detection_type = task
            process_video(video_path, video_id, detection_type)
            video_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in video processor: {str(e)}")
            video_queue.task_done()

# Start worker threads
MAX_WORKERS = 2  # Adjust based on your system's capability
for _ in range(MAX_WORKERS):
    thread = threading.Thread(target=video_processor_worker, daemon=True)
    thread.start()
    processing_threads.append(thread)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
            logger.info(f"New WebSocket connection added. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket connection removed. Total connections: {len(self.active_connections)}")
        except KeyError:
            pass

    async def broadcast(self, message: str):
        if not self.active_connections:
            logger.warning("No active connections to broadcast to")
            return

        disconnected = set()
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except WebSocketDisconnect:
                    disconnected.add(connection)
                except Exception as e:
                    logger.error(f"Error sending message: {str(e)}")
                    disconnected.add(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                self.disconnect(connection)

manager = ConnectionManager()

# Store active video streams
active_streams = {}

# Create organized storage structure
STORAGE_STRUCTURE = {
    'uploads': {
        'raw': UPLOAD_DIR / 'raw',  # Original uploaded videos
        'processed': UPLOAD_DIR / 'processed',  # Videos with detection overlays
        'thumbnails': UPLOAD_DIR / 'thumbnails',  # Video thumbnails
    },
    'detections': {
        'frames': UPLOAD_DIR / 'detections/frames',  # Individual detection frames
        'faces': UPLOAD_DIR / 'detections/faces',  # Detected faces
        'objects': UPLOAD_DIR / 'detections/objects',  # Detected objects
        'poses': UPLOAD_DIR / 'detections/poses',  # Detected poses
    }
}

# Create all directories
for category in STORAGE_STRUCTURE.values():
    for path in category.values():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")

# API Routes
@app.get("/")
async def root():
    return {"message": "Security Dashboard API is running"}

@app.get("/api/videos")
async def get_videos():
    """Get all videos and their status"""
    return await get_all_videos()

@app.get("/api/videos/{video_id}/detections")
async def get_detections(video_id: int):
    """Get all detections for a specific video"""
    return await get_detections(video_id)

@app.post("/api/videos/upload")
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
        raw_path = STORAGE_STRUCTURE['uploads']['raw'] / f"{unique_id}_{file.filename}"
        processed_path = STORAGE_STRUCTURE['uploads']['processed'] / f"{unique_id}_processed_{file.filename}"
        thumbnail_path = STORAGE_STRUCTURE['uploads']['thumbnails'] / f"{unique_id}_thumb.jpg"
        
        # Save uploaded file
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate thumbnail
        cap = cv2.VideoCapture(str(raw_path))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(thumbnail_path), frame)
        cap.release()
        
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
        
        # Broadcast upload notification
        await manager.broadcast(json.dumps({
            "type": "video_uploaded",
            "video_id": video_id,
            "filename": file.filename,
            "thumbnail_url": f"/uploads/thumbnails/{thumbnail_path.name}"
        }))
        
        # Start processing in background
        background_tasks.add_task(
            process_video_with_customer_data,
            str(raw_path),
            video_id,
            detection_type,
            str(processed_path)
        )
        
        return {
            "message": "Video uploaded and processing started",
            "video_id": video_id,
            "filename": file.filename,
            "status": "processing",
            "thumbnail_url": f"/uploads/thumbnails/{thumbnail_path.name}"
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_id}/status")
async def get_video_status(video_id: int):
    """Get the processing status of a video"""
    videos = await get_all_videos()
    video = next((v for v in videos if v["id"] == video_id), None)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return video

@app.get("/api/incidents")
async def get_all_incidents(recent: bool = Query(False)):
    """Get all incidents/alerts"""
    return await get_incidents(recent)

# Serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/processed", StaticFiles(directory="processed"), name="processed")
app.mount("/frames", StaticFiles(directory="frames"), name="frames")
app.mount("/alerts", StaticFiles(directory="alerts"), name="alerts")
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")

# Camera endpoints
@app.get("/api/cameras")
async def get_cameras():
    """Get available cameras including raw video feeds"""
    try:
        # Define absolute paths for the videos
        video_paths = [
            "E:/code/EMB Global/uploads/raw/cheese-1.mp4",
            "E:/code/EMB Global/uploads/raw/cheese-2.mp4",
            "E:/code/EMB Global/uploads/raw/Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4"
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
        
        logger.info(f"Returning cameras: {cameras}")
        return cameras
        
    except Exception as e:
        logger.error(f"Error getting cameras: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cameras/{camera_id}")
async def get_camera(camera_id: int):
    # Implement camera retrieval logic
    return {"id": camera_id, "name": "Camera Name", "status": "online"}

# Alerts endpoints
@app.get("/api/alerts")
async def get_alerts(filter: str = "recent"):
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

@app.get("/api/alerts/mall-structure")
async def get_mall_structure():
    return [
        {
            "id": 1,
            "name": "Entrance A",
            "crowd_density": 85,
            "bounds": [[0, 0], [20, 20]],
            "fill_color": "#FF4D4D"
        }
    ]

# Billing Activity endpoints
@app.get("/api/billing-activity")
async def get_billing_activity(filter: str = "all"):
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

# Customer Data endpoints
@app.get("/api/customer-data")
async def get_customer_data(
    gender: Optional[str] = None,
    date: Optional[str] = None,
    time_period: Optional[str] = None
):
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

# Daily Report endpoints
@app.get("/api/daily-report")
async def get_daily_report(date: str):
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

# System Status endpoints
@app.get("/api/system-status")
async def get_system_status():
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

# Dashboard endpoints
@app.get("/api/dashboard/statistics")
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
                "people": sum(1 for d in active_detections.values() for det in d if det["class_name"] == "person"),
                "vehicles": sum(1 for d in active_detections.values() for det in d if det["class_name"] in ["car", "truck"]),
                "objects": sum(1 for d in active_detections.values() for det in d if det["type"] == "object"),
                "faces": sum(1 for d in active_detections.values() for det in d if det["type"] == "face")
            }
        }
        
        return current_stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/incidents")
async def get_dashboard_incidents():
    return [
        {
            "id": 1,
            "title": "Unauthorized Access",
            "location": "Front Entrance",
            "time": "14:35",
            "severity": "high"
        }
    ]

@app.get("/api/dashboard/crowd-density")
async def get_crowd_density():
    return [
        {
            "id": 1,
            "name": "Entrance A",
            "density": 85,
            "status": "High"
        }
    ]

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                # Receive any messages from the client
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

# Real-time video streaming endpoint
@app.get("/api/videos/{video_id}/stream")
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

# Modified video processing function
async def process_video_with_customer_data(
    video_path: str,
    video_id: int,
    detection_type: str,
    output_path: str
):
    """Process video with real-time detection updates"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame and get detections
            detections = await process_frame(frame, detection_type)
            
            # Draw detections on frame
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save detection frames
            if detections:
                frame_path = STORAGE_STRUCTURE['detections']['frames'] / f"video_{video_id}_frame_{frame_number}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                # Store detections in database
                for detection in detections:
                    detection_data = {
                        "video_id": video_id,
                        "timestamp": datetime.now(),
                        "frame_number": frame_number,
                        "detection_type": detection["type"],
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"],
                        "class_name": detection["class_name"],
                        "image_path": str(frame_path)
                    }
                    await add_detection(detection_data)
                
                # Broadcast detection update
                await manager.broadcast(json.dumps({
                    "type": "detection_update",
                    "video_id": video_id,
                    "frame_number": frame_number,
                    "detections": detections,
                    "frame_url": f"/uploads/detections/frames/{frame_path.name}"
                }))
            
            # Write processed frame
            out.write(frame)
            frame_number += 1
            
            # Update processing progress
            progress = (frame_number / frame_count) * 100
            await manager.broadcast(json.dumps({
                "type": "processing_progress",
                "video_id": video_id,
                "progress": progress
            }))
        
        cap.release()
        out.release()
        
        # Update video status
        await update_video_status(video_id, "completed", output_path)
        
        # Broadcast completion notification
        await manager.broadcast(json.dumps({
            "type": "processing_completed",
            "video_id": video_id,
            "total_detections": len(detections),
            "output_path": output_path
        }))
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        await update_video_status(video_id, "failed")
        raise

# WebSocket endpoint for real-time dashboard updates
@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send real-time updates every second
            dashboard_data = {
                "type": "dashboard_update",
                "data": {
                    "live_feed": {
                        "people_count": 0,
                        "vehicle_count": 0,
                        "alerts": [],
                        "detections": []
                    },
                    "system_status": await get_system_status(),
                    "recent_incidents": await get_incidents(recent=True)
                }
            }
            await websocket.send_json(dashboard_data)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Live camera feed endpoint
@app.get("/api/cameras/{camera_id}/live")
async def get_live_camera_feed(camera_id: int):
    """Stream video feed with real-time processing"""
    try:
        # Get cameras to find video path
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
                        detections = await process_frame(frame)
                        
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
                            frame_path = STORAGE_STRUCTURE['detections']['frames'] / f"camera_{camera_id}_frame_{int(time.time())}.jpg"
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

@app.get("/api/detections/current")
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

async def store_detections(camera_id: int, detections: List[dict]):
    """Store detections in database"""
    try:
        async with async_session() as session:
            for det in detections:
                detection = Detection(
                    camera_id=camera_id,
                    timestamp=datetime.now(),
                    detection_type=det["class_name"],
                    confidence=det["confidence"],
                    bbox=det["bbox"],
                    frame_number=det.get("frame_number", 0)
                )
                session.add(detection)
            await session.commit()
    except Exception as e:
        logger.error(f"Error storing detections: {str(e)}")
        raise

async def get_latest_detections():
    """Get latest detections from database"""
    try:
        async with async_session() as session:
            query = select(Detection).order_by(Detection.timestamp.desc()).limit(100)
            result = await session.execute(query)
            detections = result.scalars().all()
            return [detection.to_dict() for detection in detections]
    except Exception as e:
        logger.error(f"Error getting latest detections: {str(e)}")
        return []

async def process_frame(frame, detection_type="all"):
    """Process a single frame and return detections using CUDA acceleration"""
    detections = []
    
    try:
        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to CUDA tensor for faster processing
        if torch.cuda.is_available():
            frame_tensor = torch.from_numpy(frame_rgb).cuda()
        else:
            frame_tensor = torch.from_numpy(frame_rgb)
        
        if detection_type in ["all", "face"] and face_model:
            # Process for face detection with CUDA
            face_results = face_model(frame_rgb, conf=0.5, device=0)  # device=0 for GPU
            for result in face_results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confidences):
                    detections.append({
                        "type": "face",
                        "confidence": float(conf),
                        "bbox": box.tolist(),
                        "class_name": "face"
                    })
        
        if detection_type in ["all", "object"] and object_model:
            # Process for object detection with CUDA
            object_results = object_model(frame_rgb, conf=0.5, device=0)  # device=0 for GPU
            for result in object_results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    class_name = object_model.names[int(cls_id)]
                    detections.append({
                        "type": "object",
                        "confidence": float(conf),
                        "bbox": box.tolist(),
                        "class_name": class_name
                    })
        
        return detections
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return []

@app.post("/api/videos/process-all")
async def process_all_videos(background_tasks: BackgroundTasks):
    """Process all videos in the raw folder with detections"""
    try:
        # Define video paths - only using cleaning section video for now
        videos = [
            {
                "id": 1,
                "name": "Cleaning Section",
                "path": "E:/code/EMB Global/uploads/raw/Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4"
            }
        ]

        processed_videos = []
        
        for video in videos:
            if not Path(video["path"]).exists():
                logger.warning(f"Video not found: {video['path']}")
                continue

            # Create output path for processed video
            output_filename = f"processed_{Path(video['path']).name}"
            output_path = STORAGE_STRUCTURE['uploads']['processed'] / output_filename
            
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
                process_video_batch,
                video["path"],
                video_id,
                str(output_path)
            )

        return {
            "message": "Started processing all videos",
            "videos": processed_videos
        }
        
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video_batch(video_path: str, video_id: int, output_path: str):
    """Process video in batch mode with all detections using CUDA acceleration"""
    try:
        logger.info(f"Starting batch processing for video {video_id}")
        
        # Verify models are available
        if not object_model or not face_model or not pose_model:
            error_msg = "One or more detection models not initialized"
            logger.error(error_msg)
            await update_video_status(video_id, "failed")
            await manager.broadcast(json.dumps({
                "type": "processing_error",
                "video_id": video_id,
                "error": error_msg
            }))
            raise RuntimeError(error_msg)
            
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate new dimensions maintaining aspect ratio for 720p
        target_height = 720
        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)
        
        logger.info(f"Resizing video from {width}x{height} to {target_width}x{target_height}")
        
        # Create output video writer with NVIDIA hardware acceleration
        if sys.platform == 'win32' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        if not out.isOpened():
            raise Exception("Failed to initialize video writer")
        
        frame_number = 0
        stored_detections = []
        stored_incidents = []
        last_update_time = time.time()
        last_stats_update_time = time.time()
        frames_buffer = []
        buffer_size = 32  # Reduced batch size for better memory management
        update_interval = 0.1  # Send frames more frequently (10 FPS) for smoother playback
        stats_update_interval = 2.0  # Update stats every 2 seconds
        detection_threshold = 0.4
        
        # Calculate frame skip to achieve ~15 fps processing rate
        target_processing_fps = 15
        frame_skip = max(1, round(fps / target_processing_fps))
        logger.info(f"Video FPS: {fps}, Target processing FPS: {target_processing_fps}, Frame skip: {frame_skip}")
        
        # Store original frames for output video
        original_frames_buffer = []
        
        # For UI smoothness - use a thread-safe queue for frame sending
        frame_queue = asyncio.Queue(maxsize=10)
        
        # Start a background task to send frames at a consistent rate
        send_frames_task = asyncio.create_task(send_frames_continuously(frame_queue, video_id))
        
        # For demographics tracking
        demographics = {
            "male": 0,
            "female": 0,
            "unknown": 0,
            "age_groups": {
                "child": 0,
                "young": 0,
                "adult": 0,
                "senior": 0
            }
        }
        
        # For incident tracking
        incident_trackers = {
            "loitering": {},  # person_id -> {position, start_time, frames}
            "theft": {},  # person_id -> {position, start_time, frames}
            "damage": {}  # object_id -> {position, start_time, frames}
        }
        
        # Pre-allocate CUDA tensors and optimize CUDA settings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            stream = torch.cuda.Stream()
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Process frames in batches
        while True:
            batch_start_time = time.time()
            frames_buffer = []
            original_frames_buffer = []
            processed_count = 0
            
            # Read multiple frames at once
            for _ in range(buffer_size):
                frame_read = False
                frame_to_process = None
                
                # Read frames until we find one to process
                for _ in range(frame_skip):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize frame to 720p immediately
                    orig_frame = cv2.resize(frame, (target_width, target_height))
                    
                    # Last frame in skip sequence becomes the one we process
                    frame_to_process = orig_frame.copy()
                    frame_read = True
                    frame_number += 1
                    
                    # Store all frames for output video
                    original_frames_buffer.append(orig_frame)
                
                if not frame_read:
                    break
                
                if frame_to_process is not None:
                    frames_buffer.append(frame_to_process)
                    processed_count += 1
            
            if not frames_buffer:
                break
            
            try:
                # Convert frames to RGB (already at 720p)
                frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_buffer]
                
                # Process batch with CUDA acceleration
                if torch.cuda.is_available():
                    with torch.cuda.stream(stream):
                        # Run all models on the batch
                        object_results = object_model(
                            frames_rgb,
                            conf=detection_threshold,
                            device=0,
                            batch=len(frames_buffer)
                        )
                        
                        face_results = face_model(
                            frames_rgb,
                            conf=detection_threshold,
                            device=0,
                            batch=len(frames_buffer)
                        )
                        
                        pose_results = pose_model(
                            frames_rgb,
                            conf=detection_threshold,
                            device=0,
                            batch=len(frames_buffer)
                        )
                        
                        # Process results and update frame
                        for frame_idx, (frame, obj_result, face_result, pose_result) in enumerate(zip(frames_buffer, object_results, face_results, pose_results)):
                            detections = []
                            current_frame_number = frame_number - len(frames_buffer) + frame_idx
                            
                            # Process object detections
                            if hasattr(obj_result, 'boxes'):
                                boxes = obj_result.boxes.xyxy.cpu().numpy()
                                confidences = obj_result.boxes.conf.cpu().numpy()
                                class_ids = obj_result.boxes.cls.cpu().numpy()
                                
                                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                                    class_name = object_model.names[int(cls_id)]
                                    detections.append({
                                        "type": "object",
                                        "confidence": float(conf),
                                        "bbox": box.tolist(),
                                        "class_name": class_name,
                                        "frame_number": current_frame_number
                                    })
                                    
                                    # Check for damage (e.g., broken glass, damaged property)
                                    if class_name in ["bottle", "cup", "cell phone"] and conf > 0.7:
                                        # Simple heuristic - if object is on floor
                                        x1, y1, x2, y2 = map(int, box.tolist())
                                        if y2 > target_height * 0.7:  # Object is in lower part of frame
                                            # Store as damage incident
                                            incident_id = f"damage_{class_name}_{current_frame_number}"
                                            if incident_id not in incident_trackers["damage"]:
                                                incident_trackers["damage"][incident_id] = {
                                                    "position": [(x1+x2)/2, (y1+y2)/2],
                                                    "start_time": time.time(),
                                                    "frames": [],
                                                    "class_name": class_name,
                                                    "confidence": float(conf),
                                                    "bbox": box.tolist()
                                                }
                                            
                                            # Save frame for the incident
                                            incident_trackers["damage"][incident_id]["frames"].append(frame.copy())
                                            
                                            # If we have enough frames, save as incident
                                            if len(incident_trackers["damage"][incident_id]["frames"]) >= 5:
                                                incident_name = f"damage_{class_name}_{current_frame_number}"
                                                incident_path = STORAGE_STRUCTURE['uploads']['processed'] / f"{incident_name}.jpg"
                                                cv2.imwrite(str(incident_path), frame)
                                                
                                                # Add to stored incidents
                                                stored_incidents.append({
                                                    "incident_type": "damage",
                                                    "class_name": class_name,
                                                    "confidence": float(conf),
                                                    "frame_number": current_frame_number,
                                                    "timestamp": datetime.now(),
                                                    "image_path": str(incident_path),
                                                    "video_id": video_id
                                                })
                            
                            # Process face detections
                            if hasattr(face_result, 'boxes'):
                                boxes = face_result.boxes.xyxy.cpu().numpy()
                                confidences = face_result.boxes.conf.cpu().numpy()
                                
                                for box, conf in zip(boxes, confidences):
                                    x1, y1, x2, y2 = map(int, box.tolist())
                                    
                                    # Extract face for demographic analysis
                                    face_img = frame[y1:y2, x1:x2]
                                    if face_img.size > 0:
                                        # Simulate demographic analysis (would be replaced with actual ML model)
                                        # In a real system, use a specialized model for gender/age classification
                                        face_height = y2 - y1
                                        face_width = x2 - x1
                                        
                                        # Simple heuristic based on face shape (for demo only)
                                        # Would be replaced with proper face analysis model
                                        aspect_ratio = face_width / face_height if face_height > 0 else 1
                                        face_size = face_width * face_height
                                        
                                        # Random demographic assignment for demo purposes
                                        # In production, use an actual demographic classifier
                                        gender = "male" if random.random() > 0.5 else "female"
                                        age_categories = ["child", "young", "adult", "senior"]
                                        age_group = random.choice(age_categories)
                                        
                                        # Update demographics counter
                                        demographics[gender] += 1
                                        demographics["age_groups"][age_group] += 1
                                        
                                        detections.append({
                                            "type": "face",
                                            "confidence": float(conf),
                                            "bbox": box.tolist(),
                                            "class_name": "face",
                                            "gender": gender,
                                            "age_group": age_group,
                                            "frame_number": current_frame_number
                                        })
                            
                            # Process pose detections for loitering and theft
                            if hasattr(pose_result, 'keypoints'):
                                keypoints = pose_result.keypoints.xy.cpu().numpy()
                                confidences = pose_result.keypoints.conf.cpu().numpy()
                                
                                for person_idx, (kpts, conf) in enumerate(zip(keypoints, confidences)):
                                    if len(kpts) > 0:
                                        person_id = f"person_{person_idx}_{current_frame_number}"
                                        
                                        # Get key body parts for behavioral analysis
                                        nose = kpts[0] if kpts[0][0] > 0 and kpts[0][1] > 0 else None
                                        left_wrist = kpts[9] if kpts[9][0] > 0 and kpts[9][1] > 0 else None
                                        right_wrist = kpts[10] if kpts[10][0] > 0 and kpts[10][1] > 0 else None
                                        
                                        # Calculate person center and bounding box
                                        valid_points = kpts[kpts[:, 0] > 0]
                                        if len(valid_points) > 0:
                                            person_center = np.mean(valid_points, axis=0)
                                            x_coords = valid_points[:, 0]
                                            y_coords = valid_points[:, 1]
                                            x_min, y_min = np.min(x_coords), np.min(y_coords)
                                            x_max, y_max = np.max(x_coords), np.max(y_coords)
                                            person_bbox = [x_min, y_min, x_max, y_max]
                                            
                                            # Check for loitering
                                            # Find closest existing person in tracker
                                            closest_id = None
                                            min_distance = float('inf')
                                            
                                            for tracked_id, tracked_data in incident_trackers["loitering"].items():
                                                tracked_pos = tracked_data["position"]
                                                distance = np.sqrt((tracked_pos[0] - person_center[0])**2 + 
                                                                  (tracked_pos[1] - person_center[1])**2)
                                                if distance < 50:  # Threshold for same person
                                                    if distance < min_distance:
                                                        closest_id = tracked_id
                                                        min_distance = distance
                                            
                                            if closest_id:
                                                # Update existing person
                                                duration = time.time() - incident_trackers["loitering"][closest_id]["start_time"]
                                                incident_trackers["loitering"][closest_id]["position"] = person_center
                                                incident_trackers["loitering"][closest_id]["frames"].append(frame.copy())
                                                
                                                # Check if person has been loitering
                                                if duration > 5.0:  # More than 5 seconds in same area
                                                    # Label as loitering incident
                                                    color = (0, 0, 255)  # Red color for loitering
                                                    cv2.rectangle(frame, (int(x_min), int(y_min)), 
                                                                 (int(x_max), int(y_max)), color, 2)
                                                    cv2.putText(frame, f"LOITERING {duration:.1f}s", 
                                                               (int(x_min), int(y_min)-10),
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                    
                                                    # Save incident if not already saved
                                                    if len(incident_trackers["loitering"][closest_id]["frames"]) % 30 == 0:
                                                        incident_name = f"loitering_{closest_id}_{current_frame_number}"
                                                        incident_path = STORAGE_STRUCTURE['uploads']['processed'] / f"{incident_name}.jpg"
                                                        cv2.imwrite(str(incident_path), frame)
                                                        
                                                        # Add to stored incidents
                                                        stored_incidents.append({
                                                            "incident_type": "loitering",
                                                            "duration": duration,
                                                            "frame_number": current_frame_number,
                                                            "timestamp": datetime.now(),
                                                            "image_path": str(incident_path),
                                                            "video_id": video_id
                                                        })
                                            else:
                                                # New person to track
                                                incident_trackers["loitering"][person_id] = {
                                                    "position": person_center,
                                                    "start_time": time.time(),
                                                    "frames": [frame.copy()]
                                                }
                                            
                                            # Check for theft (suspicious hand movements)
                                            if left_wrist is not None or right_wrist is not None:
                                                wrists = []
                                                if left_wrist is not None:
                                                    wrists.append(left_wrist)
                                                if right_wrist is not None:
                                                    wrists.append(right_wrist)
                                                
                                                for wrist in wrists:
                                                    # Check if hand is near shelf areas or product areas (simplified)
                                                    is_theft_area = wrist[0] < target_width*0.3 or wrist[0] > target_width*0.7
                                                    
                                                    if is_theft_area:
                                                        # Check for existing theft incident
                                                        theft_id = f"theft_{person_id}"
                                                        
                                                        if theft_id not in incident_trackers["theft"]:
                                                            incident_trackers["theft"][theft_id] = {
                                                                "position": person_center,
                                                                "start_time": time.time(),
                                                                "frames": [frame.copy()]
                                                            }
                                                        else:
                                                            duration = time.time() - incident_trackers["theft"][theft_id]["start_time"]
                                                            incident_trackers["theft"][theft_id]["frames"].append(frame.copy())
                                                            
                                                            # If suspiciously long duration in theft area
                                                            if duration > 3.0:
                                                                color = (0, 165, 255)  # Orange for theft
                                                                cv2.rectangle(frame, (int(x_min), int(y_min)), 
                                                                             (int(x_max), int(y_max)), color, 2)
                                                                cv2.putText(frame, f"SUSPICIOUS ACTIVITY {duration:.1f}s", 
                                                                           (int(x_min), int(y_min)-10),
                                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                                
                                                                # Save incident periodically
                                                                if len(incident_trackers["theft"][theft_id]["frames"]) % 30 == 0:
                                                                    incident_name = f"theft_{theft_id}_{current_frame_number}"
                                                                    incident_path = STORAGE_STRUCTURE['uploads']['processed'] / f"{incident_name}.jpg"
                                                                    cv2.imwrite(str(incident_path), frame)
                                                                    
                                                                    # Add to stored incidents
                                                                    stored_incidents.append({
                                                                        "incident_type": "theft",
                                                                        "duration": duration,
                                                                        "frame_number": current_frame_number,
                                                                        "timestamp": datetime.now(),
                                                                        "image_path": str(incident_path),
                                                                        "video_id": video_id
                                                                    })
                                        
                                        detections.append({
                                            "type": "pose",
                                            "confidence": float(conf.mean()),
                                            "keypoints": kpts.tolist(),
                                            "class_name": "person",
                                            "frame_number": current_frame_number
                                        })
                            
                            # Draw detections on frame
                            annotated_frame = frame.copy()
                            for det in detections:
                                if det["type"] in ["object", "face"]:
                                    x1, y1, x2, y2 = map(int, det["bbox"])
                                    
                                    if det["type"] == "object":
                                        color = (0, 255, 0)  # Green for objects
                                    else:
                                        color = (255, 0, 0)  # Blue for faces
                                        # Add demographic label if available
                                        if "gender" in det and "age_group" in det:
                                            demographic_label = f"{det['gender']}, {det['age_group']}"
                                            cv2.putText(annotated_frame, demographic_label, (x1, y1-30),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                                    cv2.putText(annotated_frame, label, (x1, y1-10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                elif det["type"] == "pose":
                                    for kpt in det["keypoints"]:
                                        if kpt[0] > 0 and kpt[1] > 0:
                                            x, y = map(int, kpt[:2])
                                            cv2.circle(annotated_frame, (x, y), 3, (0, 0, 255), -1)
                            
                            # Write corresponding original frames to output with annotations
                            # This ensures we maintain original FPS while processing at lower rate
                            copied_annotations = annotated_frame.copy()
                            
                            # Process corresponding frame in the buffer
                            copied_annotations = annotated_frame.copy()
                            
                            # Compress and encode the frame
                            _, buffer = cv2.imencode('.jpg', copied_annotations, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Calculate progress
                            progress = min((frame_number / frame_count) * 100, 100)
                            
                            # Prepare minimal detections data
                            minimal_detections = []
                            if len(detections) > 0:
                                minimal_detections = [
                                    {
                                        "type": det["type"],
                                        "confidence": det["confidence"],
                                        "bbox": det.get("bbox", []),
                                        "class_name": det["class_name"]
                                    } for det in detections[:5]  # Limit to 5 most important detections
                                ]
                            
                            # Add frame to the queue for consistent delivery
                            try:
                                # Don't wait if queue is full (non-blocking)
                                frame_data = {
                                    "frame": frame_base64,
                                    "detections": minimal_detections,
                                    "progress": progress
                                }
                                await frame_queue.put(frame_data)
                            except asyncio.QueueFull:
                                # Skip this frame if queue is full
                                pass
                            
                            # Update stats less frequently 
                            if time.time() - last_stats_update_time >= stats_update_interval:
                                try:
                                    await manager.broadcast(json.dumps({
                                        "type": "stats_update",
                                        "video_id": video_id,
                                        "demographics": demographics,
                                        "incidents": {
                                            "loitering": len(incident_trackers["loitering"]),
                                            "theft": len(incident_trackers["theft"]),
                                            "damage": len(incident_trackers["damage"])
                                        }
                                    }))
                                    last_stats_update_time = time.time()
                                except Exception as e:
                                    logger.error(f"Error sending stats update: {str(e)}")
                            
                            # Store detections periodically
                            if current_frame_number % 30 == 0 and detections:
                                frame_path = STORAGE_STRUCTURE['detections']['frames'] / f"video_{video_id}_frame_{current_frame_number}.jpg"
                                cv2.imwrite(str(frame_path), annotated_frame)
                                
                                stored_detections.extend([{
                                    "video_id": video_id,
                                    "frame_number": current_frame_number,
                                    "timestamp": datetime.now(),
                                    "detection_type": det["type"],
                                    "confidence": det["confidence"],
                                    "bbox": det.get("bbox", []),
                                    "keypoints": det.get("keypoints", []),
                                    "class_name": det["class_name"],
                                    "image_path": str(frame_path),
                                    "gender": det.get("gender", None),
                                    "age_group": det.get("age_group", None)
                                } for det in detections])
                    
                    # Synchronize CUDA stream
                    stream.synchronize()
                
                # Write all original frames with annotations to maintain original FPS
                for i, orig_frame in enumerate(original_frames_buffer):
                    # Check if we have annotations for this frame
                    idx_in_buffer = i % frame_skip
                    if idx_in_buffer == frame_skip - 1 and i // frame_skip < len(frames_buffer):
                        # Use the annotated frame
                        frame_to_write = frames_buffer[i // frame_skip].copy()
                    else:
                        # Use the original frame
                        frame_to_write = orig_frame.copy()
                    
                    # Write the frame
                    out.write(frame_to_write)
                
            except Exception as e:
                logger.error(f"Error processing batch at frame {frame_number}: {str(e)}")
                continue
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available() and frame_number % (buffer_size * 10) == 0:
                torch.cuda.empty_cache()
        
        # Cleanup
        cap.release()
        out.release()
        
        # Cancel the frame sending task and clean up the queue
        send_frames_task.cancel()
        try:
            await send_frames_task
        except asyncio.CancelledError:
            pass
        
        # Ensure queue is empty
        while not frame_queue.empty():
            try:
                await frame_queue.get()
                frame_queue.task_done()
            except:
                pass
        
        # Store detections in bulk
        if stored_detections:
            try:
                await add_detections_bulk(stored_detections)
            except Exception as e:
                logger.error(f"Error storing detections: {str(e)}")
        
        # Store incidents in bulk
        if stored_incidents:
            try:
                for incident in stored_incidents:
                    await add_incident(incident)
            except Exception as e:
                logger.error(f"Error storing incidents: {str(e)}")
        
        # Update video status
        await update_video_status(video_id, "completed", output_path)
        
        # Send completion message with demographics summary
        await manager.broadcast(json.dumps({
            "type": "processing_completed",
            "video_id": video_id,
            "total_detections": len(stored_detections),
            "output_path": output_path,
            "demographics": demographics,
            "incidents": {
                "loitering": len([i for i in stored_incidents if i["incident_type"] == "loitering"]),
                "theft": len([i for i in stored_incidents if i["incident_type"] == "theft"]),
                "damage": len([i for i in stored_incidents if i["incident_type"] == "damage"])
            }
        }))
        
        logger.info(f"Completed processing video {video_id}")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        await update_video_status(video_id, "failed")
        await manager.broadcast(json.dumps({
            "type": "processing_error",
            "video_id": video_id,
            "error": str(e)
        }))
        raise

# Add new endpoints for specialized detection
@app.post("/api/videos/face-extraction")
async def extract_faces(
    background_tasks: BackgroundTasks,
    video_path: str = Query(..., description="Path to input video"),
    confidence_threshold: float = Query(0.5, description="Confidence threshold for face detection")
):
    """Extract faces from video and save them"""
    try:
        # Create output directory
        save_path = STORAGE_STRUCTURE['detections']['faces'] / f"faces_{int(time.time())}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Start face extraction in background
        background_tasks.add_task(
            process_face_extraction,
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

@app.post("/api/videos/loitering-detection")
async def detect_loitering(
    background_tasks: BackgroundTasks,
    video_path: str = Query(..., description="Path to input video"),
    threshold_time: int = Query(10, description="Time threshold for loitering in seconds")
):
    """Detect loitering in video"""
    try:
        # Create output path
        output_path = STORAGE_STRUCTURE['uploads']['processed'] / f"loitering_{Path(video_path).name}"
        
        # Start loitering detection in background
        background_tasks.add_task(
            process_loitering_detection,
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

@app.post("/api/videos/theft-detection")
async def detect_theft(
    background_tasks: BackgroundTasks,
    video_path: str = Query(..., description="Path to input video"),
    hand_stay_time: int = Query(2, description="Time threshold for suspicious hand positions")
):
    """Detect suspicious behavior in video"""
    try:
        # Create output paths
        output_path = STORAGE_STRUCTURE['uploads']['processed'] / f"theft_{Path(video_path).name}"
        screenshot_dir = STORAGE_STRUCTURE['detections']['objects'] / f"theft_{int(time.time())}"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Start theft detection in background
        background_tasks.add_task(
            process_theft_detection,
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

# Add processing functions
async def process_face_extraction(video_path: str, save_path: str, confidence_threshold: float):
    """Process video for face extraction"""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run face detection
            results = face_extraction_model(frame)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    
                    if conf > confidence_threshold:
                        face_crop = frame[y1:y2, x1:x2]
                        face_filename = Path(save_path) / f"face_{frame_count}.jpg"
                        cv2.imwrite(str(face_filename), face_crop)
            
            frame_count += 1
            
        cap.release()
        logger.info(f"Face extraction completed. Saved {frame_count} frames.")
        
    except Exception as e:
        logger.error(f"Error in face extraction: {str(e)}")
        raise

async def process_loitering_detection(video_path: str, output_path: str, threshold_time: int):
    """Process video for loitering detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Tracking variables
        person_positions = {}
        next_person_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Perform object detection
            results = loitering_model(frame)
            detections = results.xyxy[0].cpu().numpy()
            
            # Track detections
            current_frame_detections = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if cls == 0 and conf > 0.5:  # Only track 'person'
                    current_frame_detections.append([x1, y1, x2, y2, conf, cls])
            
            # Assign IDs & Track Movement
            for det in current_frame_detections:
                x1, y1, x2, y2, conf, cls = det
                matched = False
                
                for person_id, data in person_positions.items():
                    last_x, last_y = data['last_position']
                    distance = np.sqrt((last_x - (x1 + x2) / 2) ** 2 + (last_y - (y1 + y2) / 2) ** 2)
                    if distance < 50:
                        person_positions[person_id]['last_position'] = ((x1 + x2) / 2, (y1 + y2) / 2)
                        matched = True
                        break
                
                if not matched:
                    person_id = next_person_id
                    next_person_id += 1
                    person_positions[person_id] = {
                        'last_position': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'time_entered': time.time()
                    }
                
                # Draw bounding box & ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {person_id}", (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Detect loitering
                time_spent = time.time() - person_positions[person_id]['time_entered']
                if time_spent > threshold_time:
                    alert_text = f"Loitering: {int(time_spent)}s"
                    cv2.putText(frame, alert_text, (int(x1), int(y1)-30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        logger.info("Loitering detection completed.")
        
    except Exception as e:
        logger.error(f"Error in loitering detection: {str(e)}")
        raise

async def process_theft_detection(video_path: str, output_path: str, screenshot_dir: str, hand_stay_time: int):
    """Process video for theft detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # COCO Pose Keypoints
        LEFT_WRIST = 9
        RIGHT_WRIST = 10
        NOSE = 0
        
        # Processing variables
        hand_timers = {}
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run pose detection
            pose_results = theft_model(frame)
            suspicious_detected = False
            
            for result in pose_results:
                keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Define regions of interest
                    chest_box = [x1 + int(0.1 * width), y1, x2 - int(0.1 * width), y1 + int(0.4 * height)]
                    left_waist_box = [x1, y1 + int(0.5 * height), x1 + int(0.5 * width), y2]
                    right_waist_box = [x1 + int(0.5 * width), y1 + int(0.5 * height), x2, y2]
                    
                    # Draw regions
                    cv2.rectangle(frame, tuple(chest_box[:2]), tuple(chest_box[2:]), (0, 255, 255), 2)
                    cv2.rectangle(frame, tuple(left_waist_box[:2]), tuple(left_waist_box[2:]), (255, 0, 0), 2)
                    cv2.rectangle(frame, tuple(right_waist_box[:2]), tuple(right_waist_box[2:]), (0, 0, 255), 2)
                    
                    if len(keypoints) <= i:
                        continue
                    
                    person_keypoints = keypoints[i]
                    left_wrist = person_keypoints[LEFT_WRIST] if person_keypoints[LEFT_WRIST][0] > 0 else None
                    right_wrist = person_keypoints[RIGHT_WRIST] if person_keypoints[RIGHT_WRIST][0] > 0 else None
                    nose = person_keypoints[NOSE] if person_keypoints[NOSE][0] > 0 else None
                    
                    # Check for suspicious behavior
                    is_back_facing = nose is None or nose[1] < y1
                    
                    for wrist, label in [(left_wrist, "left"), (right_wrist, "right")]:
                        if wrist is None:
                            continue
                        
                        wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                        
                        # Check hand positions
                        in_chest = chest_box[0] <= wrist_x <= chest_box[2] and chest_box[1] <= wrist_y <= chest_box[3]
                        in_left_waist = left_waist_box[0] <= wrist_x <= left_waist_box[2] and left_waist_box[1] <= wrist_y <= left_waist_box[3]
                        in_right_waist = right_waist_box[0] <= wrist_x <= right_waist_box[2] and right_waist_box[1] <= wrist_y <= right_waist_box[3]
                        
                        if in_chest or in_left_waist or in_right_waist:
                            if label not in hand_timers:
                                hand_timers[label] = time.time()
                            elif time.time() - hand_timers[label] > hand_stay_time:
                                suspicious_detected = True
                                cv2.putText(frame, "⚠️ Suspicious!", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                hand_timers[label] = time.time()
                        else:
                            if label in hand_timers:
                                del hand_timers[label]
                    
                    # Check back-facing behavior
                    if is_back_facing and (left_wrist is None or right_wrist is None):
                        if "back_facing" not in hand_timers:
                            hand_timers["back_facing"] = time.time()
                        elif time.time() - hand_timers["back_facing"] > hand_stay_time:
                            suspicious_detected = True
                            cv2.putText(frame, "⚠️ Suspicious (Back-Facing)", (x1, y1 - 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        if "back_facing" in hand_timers:
                            del hand_timers["back_facing"]
            
            # Save suspicious frames
            if suspicious_detected:
                screenshot_filename = Path(screenshot_dir) / f"suspicious_{frame_count}.jpg"
                cv2.imwrite(str(screenshot_filename), frame)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        logger.info("Theft detection completed.")
        
    except Exception as e:
        logger.error(f"Error in theft detection: {str(e)}")
        raise

async def send_frames_continuously(frame_queue, video_id):
    """Send frames from the queue at a consistent rate to ensure smooth playback"""
    try:
        target_fps = 10  # Target 10 FPS for smooth CCTV-like playback
        frame_interval = 1.0 / target_fps
        
        while True:
            try:
                # Get the next frame data from the queue
                frame_data = await frame_queue.get()
                if frame_data is None:  # None is our signal to stop
                    break
                
                # Send the frame through WebSocket
                await manager.broadcast(json.dumps({
                    "type": "live_detection",
                    "video_id": video_id,
                    "frame": frame_data["frame"],
                    "detections": frame_data.get("detections", []),
                    "progress": frame_data.get("progress", 0)
                }))
                
                frame_queue.task_done()
                
                # Maintain consistent frame rate
                await asyncio.sleep(frame_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending frame: {str(e)}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    except Exception as e:
        logger.error(f"Error in send_frames_continuously: {str(e)}")
    finally:
        logger.info("Frame sending task completed")

# Start the server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)