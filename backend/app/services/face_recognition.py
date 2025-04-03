import os
import logging
import uuid
import pickle
import numpy as np
import face_recognition
from datetime import datetime
from pathlib import Path

from ..core.config import settings
# from ..core.websocket import websocket_manager
from database import add_detection, add_incident

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """
    Service for advanced face recognition and tracking operations
    """
    async def track_person(self, face_image_path: str, video_path: str = None, job_id: str = None):
        """
        Track a person across a video using face recognition
        
        Args:
            face_image_path (str): Path to the reference face image
            video_path (str, optional): Path to the video to search in
            job_id (str, optional): Unique job identifier
        
        Returns:
            Dict: Tracking results
        """
        try:
            # Generate unique identifiers
            job_id = job_id or f"tracking_{uuid.uuid4().hex[:8]}"
            screenshot_dir = os.path.join(settings.SCREENSHOTS_DIR, f"face_tracking_{job_id}")
            os.makedirs(screenshot_dir, exist_ok=True)
            
            # Generate output paths
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                settings.PROCESSED_DIR, 
                f"tracked_{timestamp}_{Path(video_path).name if video_path else 'unknown'}"
            )
            
            # Load reference face
            query_image = face_recognition.load_image_file(face_image_path)
            query_face_encodings = face_recognition.face_encodings(query_image)
            
            if not query_face_encodings:
                logger.error(f"No face detected in image: {face_image_path}")
                return {
                    "status": "error",
                    "message": "No face detected in reference image",
                    "job_id": job_id
                }
            
            # Get the first face encoding
            query_encoding = query_face_encodings[0]
            
            # Open video
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Tracking variables
            frame_count = 0
            matches = []
            screenshots = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame to improve performance
                if frame_count % 5 == 0:
                    # Convert to RGB for face recognition
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces in the frame
                    face_locations = face_recognition.face_locations(frame_rgb)
                    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
                    
                    # Compare faces
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Compare face with query face
                        face_distances = face_recognition.face_distance([query_encoding], face_encoding)
                        
                        # If match found
                        if face_distances[0] < 0.6:  # Lower is more similar
                            confidence = 1.0 - face_distances[0]
                            
                            # Draw rectangle around matched face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            # Add confidence text
                            label = f"Match: {confidence:.2f}"
                            cv2.putText(frame, label, (left, top - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            # Track detection
                            match_data = {
                                "frame": frame_count,
                                "timestamp": frame_count / fps,
                                "confidence": float(confidence),
                                "bbox": [left, top, right, bottom]
                            }
                            matches.append(match_data)
                            
                            # Periodically save screenshots
                            if len(matches) % 5 == 0:
                                screenshot_path = os.path.join(
                                    screenshot_dir, 
                                    f"match_{frame_count:05d}.jpg"
                                )
                                cv2.imwrite(screenshot_path, frame)
                                screenshots.append({
                                    "path": screenshot_path,
                                    "frame": frame_count,
                                    "confidence": float(confidence)
                                })
                    
                    # Update via WebSocket periodically
                    
                
                # Write processed frame
                out.write(frame)
                frame_count += 1
            
            # Release resources
            cap.release()
            out.release()
            
            # Prepare results
            results = {
                "job_id": job_id,
                "status": "completed",
                "output_path": output_path,
                "total_frames": frame_count,
                "detections": matches,
                "screenshots": screenshots
            }
            
            # Save results for later retrieval
            results_path = os.path.join(settings.PROCESSED_DIR, f"{job_id}_results.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            # Broadcast completion
            
            return results
        
        except Exception as e:
            logger.error(f"Face tracking error: {str(e)}")
            
            # Broadcast error
         
            
            return {
                "status": "error",
                "message": str(e),
                "job_id": job_id
            }

# Singleton service instance
face_recognition_service = FaceRecognitionService()