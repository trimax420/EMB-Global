from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from typing import Optional, List
import cv2
import numpy as np
import os
import aiofiles
from datetime import datetime
import time
import logging
from pathlib import Path
import json
import pickle
import uuid

from ...core.config import settings
from ...core.websocket import manager
from ...services.video_processor import video_processor
from database import add_detection, add_incident, get_customer_data

router = APIRouter()
logger = logging.getLogger(__name__)

# Directory to store uploaded face images
FACE_UPLOADS_DIR = os.path.join(settings.UPLOAD_DIR, "faces")
os.makedirs(FACE_UPLOADS_DIR, exist_ok=True)

@router.post("/track-person")
async def track_person(
    background_tasks: BackgroundTasks,
    face_image: UploadFile = File(...),
    video_path: str = Query(..., description="Path to video to search in"),
    similarity_threshold: float = Query(0.6, description="Threshold for face matching (0-1, lower is stricter)"),
    skip_frames: int = Query(5, description="Process every Nth frame to improve performance")
):
    """
    Track a person throughout a video feed based on their face image.
    Returns timestamps and locations where the person appears.
    
    Args:
        face_image: Uploaded image of the face to search for
        video_path: Path to the video to search within
        similarity_threshold: Threshold for face matching (0-1)
        skip_frames: Process every Nth frame (for performance)
    
    Returns:
        Tracking job information and initial results if available
    """
    try:
        # Validate the face image
        if not face_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename and save the face image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_id = f"face_{uuid.uuid4().hex[:8]}"
        face_filename = f"{timestamp}_{face_id}_{face_image.filename}"
        face_path = os.path.join(FACE_UPLOADS_DIR, face_filename)
        
        # Save the face image
        async with aiofiles.open(face_path, 'wb') as out_file:
            content = await face_image.read()
            await out_file.write(content)
        
        # Generate path for output video with tracking visualization
        output_filename = f"tracked_{timestamp}_{Path(video_path).name}"
        output_path = os.path.join(settings.PROCESSED_DIR, output_filename)
        
        # Directory for saving screenshots when person is detected
        screenshot_dir = os.path.join(settings.SCREENSHOTS_DIR, f"person_tracking_{timestamp}")
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Start person tracking in background
        tracking_job_id = f"tracking_{uuid.uuid4().hex[:8]}"
        background_tasks.add_task(
            process_person_tracking,
            face_path,
            video_path,
            output_path,
            screenshot_dir,
            similarity_threshold,
            skip_frames,
            tracking_job_id
        )
        
        return {
            "message": "Person tracking started",
            "job_id": tracking_job_id,
            "face_id": face_id,
            "face_path": face_path,
            "output_path": output_path,
            "screenshot_dir": screenshot_dir,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error starting person tracking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tracking-results/{job_id}")
async def get_tracking_results(job_id: str):
    """
    Get the results of a tracking job.
    
    Args:
        job_id: ID of the tracking job
        
    Returns:
        Current tracking results
    """
    try:
        # Check if results file exists
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        
        if not os.path.exists(results_path):
            return {
                "status": "processing",
                "message": "Tracking results not available yet",
                "detections": []
            }
        
        # Load results
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        return {
            "status": results.get("status", "processing"),
            "message": results.get("message", "Processing tracking results"),
            "progress": results.get("progress", 0),
            "detections": results.get("detections", []),
            "total_frames_processed": results.get("frames_processed", 0),
            "total_matches": results.get("total_matches", 0),
            "output_video_path": results.get("output_path", ""),
            "screenshots": results.get("screenshots", [])
        }
        
    except Exception as e:
        logger.error(f"Error retrieving tracking results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customer-history")
async def get_customer_history(face_image_path: str):
    """
    Search for a customer across all stored customer data using face matching.
    Allows looking up if a customer has been in the store before.
    
    Args:
        face_image_path: Path to the face image to search for
        
    Returns:
        History of customer appearances if found
    """
    try:
        # Load the query face
        if not os.path.exists(face_image_path):
            raise HTTPException(status_code=404, detail="Face image not found")
            
        # Get face encoding
        import face_recognition
        query_image = face_recognition.load_image_file(face_image_path)
        query_face_encodings = face_recognition.face_encodings(query_image)
        
        if not query_face_encodings:
            raise HTTPException(status_code=400, detail="No face detected in the provided image")
        
        query_encoding = query_face_encodings[0]
        
        # Get all customer data
        all_customers = await get_customer_data()
        matched_customers = []
        
        # Search for matches
        for customer in all_customers:
            # Skip customers without face encoding
            if not customer.get("face_encoding"):
                continue
                
            # Compare face encodings
            try:
                customer_encoding = np.array(customer["face_encoding"])
                distance = face_recognition.face_distance([customer_encoding], query_encoding)[0]
                
                # Consider a match if distance is below threshold
                if distance < 0.6:  # Lower values are better matches
                    matched_customers.append({
                        "customer_id": customer["id"],
                        "match_confidence": float(1.0 - distance),  # Convert to similarity score
                        "entry_time": customer["entry_time"],
                        "entry_date": customer["entry_date"],
                        "image_url": customer["image_url"],
                        "gender": customer.get("gender", "unknown"),
                        "age_group": customer.get("age_group", "unknown")
                    })
            except Exception as e:
                logger.warning(f"Error comparing face encodings for customer {customer['id']}: {str(e)}")
        
        # Sort by confidence
        matched_customers.sort(key=lambda x: x["match_confidence"], reverse=True)
        
        return {
            "query_face": face_image_path,
            "total_matches": len(matched_customers),
            "matched_customers": matched_customers
        }
        
    except Exception as e:
        logger.error(f"Error searching customer history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_person_tracking(face_image_path, video_path, output_path, screenshot_dir, 
                                 similarity_threshold, skip_frames, job_id):
    """
    Process video to track a person based on their face image.
    
    Args:
        face_image_path: Path to face image to search for
        video_path: Path to video to search in
        output_path: Path to save processed video with tracking visualization
        screenshot_dir: Directory to save detection screenshots
        similarity_threshold: Threshold for face matching (0-1)
        skip_frames: Process every Nth frame
        job_id: Unique ID for this tracking job
    """
    try:
        logger.info(f"Starting person tracking job {job_id} for {video_path}")
        
        # Load the face_recognition module
        import face_recognition
        
        # Load the query face
        query_image = face_recognition.load_image_file(face_image_path)
        query_face_encodings = face_recognition.face_encodings(query_image)
        
        if not query_face_encodings:
            raise ValueError("No face detected in the provided image")
        
        query_encoding = query_face_encodings[0]
        
        # Initialize results dictionary
        results = {
            "status": "processing",
            "message": "Starting video processing",
            "progress": 0,
            "detections": [],
            "frames_processed": 0,
            "total_matches": 0,
            "output_path": output_path,
            "screenshots": []
        }
        
        # Save initial results
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        matches_found = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every Nth frame to improve performance
            if frame_count % skip_frames == 0:
                # Update progress
                progress = min(100, int((frame_count / total_frames) * 100))
                results["progress"] = progress
                results["frames_processed"] = frame_count
                
                # Detect faces in the current frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(frame_rgb)
                
                if face_locations:
                    # Get encodings for all faces in the frame
                    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
                    
                    # Compare each face with the query face
                    for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
                        face_distance = face_recognition.face_distance([query_encoding], encoding)[0]
                        
                        # Match found if distance is below threshold
                        if face_distance < similarity_threshold:
                            matches_found += 1
                            
                            # Get face location
                            top, right, bottom, left = location
                            
                            # Draw rectangle around matched face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            # Add confidence score
                            confidence = 1.0 - face_distance
                            conf_text = f"Match: {confidence:.2f}"
                            cv2.putText(frame, conf_text, (left, top - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Calculate timestamp
                            timestamp = frame_count / fps
                            mins = int(timestamp // 60)
                            secs = int(timestamp % 60)
                            timestamp_str = f"{mins:02d}:{secs:02d}"
                            
                            # Save screenshot
                            if matches_found % 5 == 0:  # Save every 5th match to avoid too many screenshots
                                screenshot_path = os.path.join(screenshot_dir, f"match_{matches_found:03d}_{timestamp_str}.jpg")
                                cv2.imwrite(screenshot_path, frame)
                                
                                # Add to screenshots list
                                results["screenshots"].append({
                                    "path": screenshot_path,
                                    "timestamp": timestamp,
                                    "timestamp_str": timestamp_str,
                                    "confidence": float(confidence),
                                    "location": {"top": top, "right": right, "bottom": bottom, "left": left}
                                })
                            
                            # Add to detections list
                            results["detections"].append({
                                "frame": frame_count,
                                "timestamp": timestamp,
                                "timestamp_str": timestamp_str,
                                "confidence": float(confidence),
                                "location": {"top": top, "right": right, "bottom": bottom, "left": left}
                            })
                            
                            # Add detection to database for tracking
                            try:
                                detection_data = {
                                    "video_id": None,  # For tracking, we may not have a video ID
                                    "timestamp": datetime.now(),
                                    "frame_number": frame_count,
                                    "detection_type": "face_tracking",
                                    "confidence": float(confidence),
                                    "bbox": [left, top, right, bottom],
                                    "class_name": "person",
                                    "image_path": screenshot_path if matches_found % 5 == 0 else "",
                                    "detection_metadata": {
                                        "tracking_job_id": job_id,
                                        "timestamp_str": timestamp_str,
                                        "face_image": face_image_path
                                    }
                                }
                                await add_detection(detection_data)
                            except Exception as e:
                                logger.error(f"Error adding detection to database: {str(e)}")
                
                # Update results periodically
                if frame_count % (fps * 5) == 0:  # Update every 5 seconds of video
                    results["total_matches"] = matches_found
                    with open(results_path, 'wb') as f:
                        pickle.dump(results, f)
                    
                    # Update via WebSocket if needed
                    await manager.broadcast(json.dumps({
                        "type": "tracking_progress", 
                        "job_id": job_id, 
                        "progress": progress, 
                        "matches": matches_found
                    }))
            
            # Write frame to output video
            out.write(frame)
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        # Final update to results
        results["status"] = "completed"
        results["message"] = "Person tracking completed"
        results["progress"] = 100
        results["frames_processed"] = frame_count
        results["total_matches"] = matches_found
        
        # Save final results
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Send completion notification
        await manager.broadcast(json.dumps({
            "type": "tracking_completed", 
            "job_id": job_id, 
            "matches": matches_found,
            "output_path": output_path
        }))
        
        logger.info(f"Person tracking job {job_id} completed. Found {matches_found} matches.")
        
        # Create an incident if matches were found (optional)
        if matches_found > 0:
            try:
                incident_data = {
                    "type": "person_tracking",
                    "timestamp": datetime.now(),
                    "location": "Security camera",
                    "description": f"Person of interest tracked in video with {matches_found} appearances",
                    "image_path": results["screenshots"][0]["path"] if results["screenshots"] else face_image_path,
                    "video_url": output_path,
                    "severity": "medium",
                    "confidence": results["detections"][0]["confidence"] if results["detections"] else 0.7,
                    "is_resolved": False
                }
                incident_id = await add_incident(incident_data)
                logger.info(f"Created tracking incident with ID {incident_id}")
            except Exception as e:
                logger.error(f"Error creating tracking incident: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in person tracking job {job_id}: {str(e)}")
        
        # Update results with error
        results = {
            "status": "failed",
            "message": f"Error processing video: {str(e)}",
            "progress": 0,
            "detections": [],
            "frames_processed": frame_count if 'frame_count' in locals() else 0,
            "total_matches": matches_found if 'matches_found' in locals() else 0,
            "output_path": output_path,
            "screenshots": []
        }
        
        # Save error results
        results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Send error notification
        await manager.broadcast(json.dumps({
            "type": "tracking_failed", 
            "job_id": job_id, 
            "error": str(e)
        }))