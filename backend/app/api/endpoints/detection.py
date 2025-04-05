from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
import cv2
import numpy as np
import tempfile
import os
import time
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.schemas.detection import DetectionCreate, Detection
from app.services.detector import DetectionService
from app.services.tracker import PersonTracker
from app.services.behavior import BehaviorAnalysisService
from app.services.alert import AlertService
from app.models.detection import DetectionEvent
from app.models.alert import Alert
from app.utils.video import process_video_file

router = APIRouter()

# Singleton services
detector_service = DetectionService()
tracker_service = PersonTracker()
behavior_service = BehaviorAnalysisService()
alert_service = AlertService()

@router.post("/process-video/", response_model=dict)
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Process a video file for detection analysis
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    # Add video processing task to background tasks
    job_id = f"video_process_{int(time.time())}"
    background_tasks.add_task(
        process_video_file,
        file_path=temp_file_path,
        job_id=job_id,
        db=db
    )
    
    return {"job_id": job_id, "status": "Processing started"}

@router.get("/job-status/{job_id}", response_model=dict)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Get status of a video processing job
    """
    # In a real implementation, check job status in the database
    # This is a simplified example
    return {"job_id": job_id, "status": "processing"}

@router.get("/detections/", response_model=List[Detection])
async def get_detections(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    zone_id: Optional[int] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get detection events with optional filtering
    """
    query = db.query(DetectionEvent)
    
    if start_time:
        query = query.filter(DetectionEvent.timestamp >= start_time)
    if end_time:
        query = query.filter(DetectionEvent.timestamp <= end_time)
    if zone_id:
        # This assumes zone_id is stored in detection_data JSON
        query = query.filter(DetectionEvent.detection_data.contains({"zone_id": zone_id}))
    
    events = query.order_by(DetectionEvent.timestamp.desc()).limit(limit).all()
    return events