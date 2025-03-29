from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

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