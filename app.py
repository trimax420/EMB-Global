from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Query, Path, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional, Any
import os
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
from typing import Union
from database import (
    init_db,
    get_all_videos,
    get_video_detections,
    get_incidents,
    add_video,
    add_detection,
    add_incident,
    update_video_status
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
FACE_MODEL_PATH = MODELS_DIR / "yolov8n-face.pt"
POSE_MODEL_PATH = MODELS_DIR / "yolov8n-pose.pt"
OBJECT_MODEL_PATH = MODELS_DIR / "yolov5sn.pt"

# Load models if YOLO is available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLO package is available")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO package is not available")

face_model = None
pose_model = None
object_model = None

if YOLO_AVAILABLE:
    try:
        if FACE_MODEL_PATH.exists():
            face_model = YOLO(str(FACE_MODEL_PATH))
            logger.info("Face detection model loaded successfully")
        else:
            logger.warning(f"Face model not found at {FACE_MODEL_PATH}")

        if POSE_MODEL_PATH.exists():
            pose_model = YOLO(str(POSE_MODEL_PATH))
            logger.info("Pose detection model loaded successfully")
        else:
            logger.warning(f"Pose model not found at {POSE_MODEL_PATH}")

        if OBJECT_MODEL_PATH.exists():
            object_model = YOLO(str(OBJECT_MODEL_PATH))
            logger.info("Object detection model loaded successfully")
        else:
            logger.warning(f"Object model not found at {OBJECT_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

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
    detection_ids: List[int]

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
    return await get_video_detections(video_id)

@app.post("/api/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detection_type: str = Query(..., description="Type of detection: theft, loitering, face_detection")
):
    """Upload video for processing"""
    if detection_type not in ["theft", "loitering", "face_detection"]:
        raise HTTPException(status_code=400, detail="Invalid detection type")
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    upload_path = UPLOAD_DIR / f"{unique_id}_{file.filename}"
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Add video to database
    video_data = {
        "name": file.filename,
        "file_path": str(upload_path),
        "status": "pending",
        "detection_type": detection_type,
        "upload_time": datetime.now()
    }
    video_id = await add_video(video_data)
    
    # Add to processing queue
    video_queue.put((str(upload_path), video_id, detection_type))
    
    return {
        "message": "Video uploaded and queued for processing",
        "video_id": video_id,
        "filename": file.filename,
        "status": "pending"
    }

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
    cameras = [
        {
            "id": 1,
            "name": "Front Entrance",
            "status": "online",
            "stream_url": "https://developer-blogs.nvidia.com/wp-content/uploads/2022/12/Figure8-output_blurred-compressed.gif"
        },
        {
            "id": 2,
            "name": "Main Hall",
            "status": "online",
            "stream_url": "https://user-images.githubusercontent.com/11428131/139924111-58637f2e-f2f6-42d8-8812-ab42fece92b4.gif"
        }
    ]
    return cameras

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
    return {
        "total_cameras": 12,
        "active_detections": 25,
        "current_alerts": 7,
        "system_status": "Optimal"
    }

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

# Start the server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)