from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
import os
import aiofiles
from datetime import datetime

router = APIRouter()

UPLOAD_DIR = "uploads/videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    detection_type: str = Query(..., enum=["theft", "loitering", "face_detection"])
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{video.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Save the video file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await video.read()
            await out_file.write(content)
            
        # Here you would typically start your video processing pipeline
        # For now, we'll just return success
        return {
            "message": "Video uploaded successfully",
            "filename": filename,
            "detection_type": detection_type,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 