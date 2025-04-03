from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class CreateCameraRequest(BaseModel):
    """Request model for creating a new camera"""
    name: str = Field(..., description="Camera name")
    location: str = Field(..., description="Camera location")
    url: Optional[str] = Field(None, description="RTSP stream URL for network cameras")
    camera_id: Optional[int] = Field(None, description="Camera ID for local USB cameras")
    video_path: Optional[str] = Field(None, description="Path to video file for file-based cameras")
    resolution: Optional[Dict[str, int]] = Field(None, description="Camera resolution (width, height)")
    capabilities: Optional[List[str]] = Field(None, description="Camera capabilities (e.g., face_detection, theft_detection)")
    
class Camera(BaseModel):
    """Camera model"""
    id: int = Field(..., description="Camera ID")
    name: str = Field(..., description="Camera name")
    location: str = Field(..., description="Camera location")
    url: Optional[str] = Field(None, description="RTSP stream URL for network cameras")
    camera_id: Optional[int] = Field(None, description="Camera ID for local USB cameras")
    video_path: Optional[str] = Field(None, description="Path to video file for file-based cameras")
    resolution: Optional[Dict[str, int]] = Field(None, description="Camera resolution (width, height)")
    capabilities: Optional[List[str]] = Field(None, description="Camera capabilities")
    status: str = Field("offline", description="Camera status (online, offline, error)")
    last_seen: Optional[str] = Field(None, description="Timestamp of last activity")
    error: Optional[str] = Field(None, description="Error message if status is error")
    
    class Config:
        orm_mode = True 