import os
import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from ..core.config import settings
from .detection_service import DetectorService

logger = logging.getLogger(__name__)

class MLService:
    """Service for managing machine learning operations and inference."""
    
    def __init__(self):
        """Initialize the ML service"""
        self.detector = DetectorService.get_instance()
        
    async def process_frame(self, frame: np.ndarray, detection_type: str = "all") -> List[Dict]:
        """
        Process a single frame through appropriate ML models based on detection type.
        
        Args:
            frame: Input video frame as numpy array
            detection_type: Type of detection to perform (all, theft, loitering, face)
            
        Returns:
            List of detection results with bounding boxes, classes, etc.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received for processing")
            return []
            
        try:
            detections = []
            
            # Route to appropriate detection method based on type
            if detection_type == "all":
                # Detect objects
                object_results = await self.detector.detect_objects(frame)
                detections.extend(object_results)
                
                # Detect people
                people_results = await self.detector.detect_people(frame)
                detections.extend(people_results)
                
                # We'd add more detections here in a complete implementation
                
            elif detection_type == "theft":
                theft_results = await self.detector.detect_theft(frame)
                detections.extend(theft_results)
                
            elif detection_type == "loitering":
                # For loitering, we'd typically track people over time
                # This would require state across frames, but for this placeholder:
                loitering_results = await self.detector.detect_loitering(frame, [], {})
                detections.extend(loitering_results)
                
            elif detection_type == "face":
                # In a real implementation, this would use face detection models
                # For simplicity in this placeholder, we'll use the people detector
                face_results = await self.detector.detect_people(frame)
                for result in face_results:
                    result["type"] = "face"
                detections.extend(face_results)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return []
    
    async def process_video(self, 
                           video_path: str, 
                           output_path: Optional[str] = None,
                           detection_type: str = "all",
                           start_frame: int = 0,
                           max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a video file with ML detection.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            detection_type: Type of detection to perform
            start_frame: Frame to start processing from
            max_frames: Maximum number of frames to process (optional)
            
        Returns:
            Dict with processing results including detections
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Set default output path if not provided
        if output_path is None:
            basename = os.path.basename(video_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(settings.PROCESSED_DIR, f"{name}_processed{ext}")
            
        results = {
            "input_path": video_path,
            "output_path": output_path,
            "detections": [],
            "frames_processed": 0,
            "detection_type": detection_type
        }
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Set up video writer if output path is provided
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            else:
                out = None
                
            # Skip to start frame if specified
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
            # Calculate end frame if max_frames is specified
            end_frame = total_frames
            if max_frames is not None:
                end_frame = min(start_frame + max_frames, total_frames)
                
            frame_count = 0
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames is not None and frame_count >= max_frames):
                    break
                    
                # Get current frame number
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # Process every Nth frame for efficiency (process all frames for this placeholder)
                if frame_count % 1 == 0:
                    # Process frame
                    detections = await self.process_frame(frame, detection_type)
                    
                    # Add frame number to detections
                    for detection in detections:
                        detection["frame_number"] = current_frame
                        results["detections"].append(detection)
                        
                    # Draw detections on frame for output video
                    if out is not None:
                        for det in detections:
                            bbox = det.get("bbox", [])
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = map(int, bbox)
                                
                                # Select color based on detection type
                                color = (0, 255, 0)  # Default green
                                if det.get("type") == "theft":
                                    color = (0, 0, 255)  # Red
                                elif det.get("type") == "loitering":
                                    color = (0, 165, 255)  # Orange
                                    
                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Add label
                                label = f"{det.get('class_name', 'object')} {det.get('confidence', 0.0):.2f}"
                                cv2.putText(frame, label, (x1, y1 - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                        # Write frame to output video
                        out.write(frame)
                        
                frame_count += 1
                
                # Update results
                results["frames_processed"] = frame_count
                
                # Log progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing video: {progress:.1f}% complete")
                    
            # Release resources
            cap.release()
            if out is not None:
                out.release()
                
            # Success
            logger.info(f"Video processing complete: {results['frames_processed']} frames processed")
            results["status"] = "success"
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
            return results 