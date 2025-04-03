import os
# Force torch to not load weights only
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import cv2
import time
import logging
import numpy as np
import asyncio
from datetime import datetime
import math
import hashlib
import random
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ..core.config import settings
from ..core.websocket import websocket_manager
import face_recognition
import pickle
import uuid
from database import update_customer_face_encoding

logger = logging.getLogger(__name__)

class VideoFileCamera:
    """Class to handle video files as camera feeds"""
    def __init__(self, camera_id, video_dir=None, specific_video=None):
        self.camera_id = str(camera_id)
        # Try to get video directory from environment or settings
        if video_dir:
            self.video_dir = video_dir
        else:
            # Use environment variable directly if set
            env_dir = os.getenv("VIDEO_FILES_DIR")
            if env_dir:
                self.video_dir = env_dir
                logger.info(f"Using video directory from environment: {self.video_dir}")
            else:
                # Fall back to settings
                self.video_dir = settings.VIDEO_FILES_DIR
                logger.info(f"Using video directory from settings: {self.video_dir}")
        
        logger.info(f"Initializing VideoFileCamera for camera {camera_id} with directory {self.video_dir}")
        
        # If a specific video is provided, use only that one
        if specific_video:
            if os.path.exists(specific_video):
                self.video_files = [specific_video]
                logger.info(f"Using specific video file: {specific_video}")
            else:
                logger.error(f"Specified video file not found: {specific_video}")
                self.video_files = []
        else:
            self.video_files = self._find_video_files()
            
        self.current_file_index = 0
        self.cap = None
        self.current_filename = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.current_frame = 0
        self.loop = True  # Whether to loop videos
        
        # Initialize the video capture
        if self.video_files:
            self.init_video_capture()
        else:
            logger.error(f"No video files available for camera {camera_id}")
        
    def _find_video_files(self):
        """Find all video files in the video directory"""
        logger.info(f"Looking for video files in: {self.video_dir}")
        if not os.path.exists(self.video_dir):
            logger.warning(f"Video directory does not exist: {self.video_dir}")
            return []
        
        video_files = []
        # Get all video files with common video extensions
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]:
            video_files.extend(glob.glob(os.path.join(self.video_dir, ext)))
        
        # Try to find a specific video based on camera ID
        if video_files:
            # Default video mappings for specific camera IDs
            camera_video_mapping = {
                "1": "cheese_store_NVR_2_NVR_2_20250214160950_20250214161659_844965939.mp4",
                "2": "cheese-1.mp4",
                "3": "cheese-2.mp4",
                "4": "Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4"
            }
            
            # First check if there's an environment variable override
            env_source = os.getenv(f"VIDEO_SOURCE_{self.camera_id}")
            if not env_source and self.camera_id in camera_video_mapping:
                # Use the default mapping if no environment variable
                env_source = camera_video_mapping[self.camera_id]
                
            if env_source:
                # Check both for the filename directly and with full path
                full_path = os.path.join(self.video_dir, env_source)
                if os.path.exists(full_path):
                    logger.info(f"Using specific video for camera {self.camera_id}: {env_source}")
                    # This is the assigned video for this camera
                    return [full_path]  # Return only this specific video
                else:
                    logger.warning(f"Mapped video file not found: {full_path}")
            
            # If we get to this point, either no mapping exists or the mapped file wasn't found
            # For numbered camera IDs (1-4), we can use modulo to assign different videos
            if self.camera_id.isdigit():
                camera_num = int(self.camera_id)
                if 1 <= camera_num <= 4 and len(video_files) > 0:
                    # Use a specific video for this camera based on its number
                    idx = (camera_num - 1) % len(video_files)
                    specific_file = video_files[idx]
                    logger.info(f"Assigned video #{idx+1} to camera {self.camera_id}: {os.path.basename(specific_file)}")
                    return [specific_file]  # Return only this specific video
        
        # If we couldn't find a specific video for this camera, return all videos
        if not video_files:
            logger.warning(f"No video files found in {self.video_dir}")
        else:
            logger.info(f"Found {len(video_files)} video files")
            
        return video_files
    
    def init_video_capture(self):
        """Initialize video capture from available video files"""
        if not self.video_files:
            logger.error("No video files available to initialize capture")
            return False
            
        video_path = self.video_files[self.current_file_index]
        self.current_filename = os.path.basename(video_path)
        
        # Release previous capture if it exists
        if self.cap is not None:
            self.cap.release()
            
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return False
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        logger.info(f"Video loaded: {self.current_filename}, "
                  f"FPS: {self.fps}, Resolution: {self.width}x{self.height}, "
                  f"Total frames: {self.total_frames}")
        return True
    
    def read(self):
        """Read a frame from the video file"""
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Video capture not initialized")
            return False, None
        
        # Read frame
        ret, frame = self.cap.read()
        
        # If reached end of video
        if not ret:
            if self.loop:
                # Try to move to next video file
                self.current_file_index = (self.current_file_index + 1) % len(self.video_files)
                logger.info(f"Moving to next video: {self.video_files[self.current_file_index]}")
                
                # Initialize with the next video
                success = self.init_video_capture()
                if success:
                    # Read frame from new video
                    ret, frame = self.cap.read()
                else:
                    return False, None
            else:
                # Don't loop, just return failure
                return False, None
        
        # Increment frame counter
        self.current_frame += 1
        
        return ret, frame
    
    def release(self):
        """Release the video capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"Released video file: {self.current_filename}")
    
    def get_info(self):
        """Get information about the video file"""
        return {
            "camera_id": self.camera_id,
            "filename": self.current_filename,
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "progress": round((self.current_frame / max(1, self.total_frames)) * 100, 2),
            "file_index": f"{self.current_file_index + 1}/{len(self.video_files)}"
        }

class VideoProcessor:
    """Service for processing video streams and applying ML models"""
    
    def __init__(self):
        # Store video file cameras for reuse
        self.video_cameras = {}
        
        # Cache for mock frame generation
        self._mock_frames = {}
        
        # Initialize inference models
        self.theft_detection_model = TheftDetectionModel()
        self.loitering_detection_model = LoiteringDetectionModel()
        
        logger.info("VideoProcessor initialized")
        
    async def get_video_frame(self, camera_id, use_mock=True, use_video=True):
        """Get a frame from a video file or mock data for the given camera ID"""
        frame = None
        is_mock = False
        video_info = None
        is_video_file = False
        
        # Try to get frame from video file first if enabled
        if use_video:
            # Check if we already have a video camera for this ID
            if camera_id not in self.video_cameras:
                # Create a new video camera
                self.video_cameras[camera_id] = VideoFileCamera(camera_id)
            
            # Read a frame from the video camera
            ret, frame = self.video_cameras[camera_id].read()
            
            if ret and frame is not None:
                is_video_file = True
                video_info = self.video_cameras[camera_id].get_info()
        
        # If we couldn't get a frame from video file or it's not enabled, try mock data
        if frame is None and use_mock:
            frame = self.get_mock_frame(camera_id)
            is_mock = True
            
        return frame, is_mock, video_info, is_video_file
        
    def release_video_cameras(self, camera_id=None):
        """Release video cameras"""
        if camera_id is not None and camera_id in self.video_cameras:
            # Release specific camera
            self.video_cameras[camera_id].release()
            del self.video_cameras[camera_id]
            logger.info(f"Released video camera for camera ID {camera_id}")
        else:
            # Release all cameras
            for camera_id, camera in self.video_cameras.items():
                camera.release()
            self.video_cameras = {}
            logger.info("Released all video cameras")
    
    def get_mock_frame(self, camera_id):
        """Generate a mock camera frame with a test pattern"""
        # Check if we have a cached mock frame for this camera
        if camera_id in self._mock_frames:
            mock_frame = self._mock_frames[camera_id].copy()
        else:
            # Generate a new mock frame
            mock_frame = np.zeros((settings.MOCK_FRAME_HEIGHT, settings.MOCK_FRAME_WIDTH, 3), dtype=np.uint8)
            
            # Fill with a gray background
            mock_frame[:] = (200, 200, 200)
            
            # Add an identification pattern based on camera_id
            id_value = int(hashlib.md5(camera_id.encode()).hexdigest(), 16) % 255
            
            # Create a test pattern with colored bars
            bar_width = settings.MOCK_FRAME_WIDTH // 8
            for i in range(8):
                c = (i * 30 + id_value) % 255
                start_x = i * bar_width
                end_x = start_x + bar_width
                color = [(c+80)%255, (c+160)%255, (c+240)%255]
                mock_frame[:, start_x:end_x] = color
            
            # Add text with camera ID and timestamp
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(mock_frame, f"Camera {camera_id} (Mock)", (20, 40), font, 0.8, (0, 0, 0), 2)
            
            # Cache the base mock frame
            self._mock_frames[camera_id] = mock_frame.copy()
            
        # Add dynamic elements (timestamp, counter)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(mock_frame, timestamp, (20, settings.MOCK_FRAME_HEIGHT - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                   
        # Add moving element for visual interest
        seconds = time.time()
        x = int((settings.MOCK_FRAME_WIDTH - 50) * (0.5 + 0.5 * math.sin(seconds)))
        y = int((settings.MOCK_FRAME_HEIGHT - 50) * (0.5 + 0.5 * math.cos(seconds * 0.5)))
        cv2.circle(mock_frame, (x, y), 20, (0, 0, 255), -1)
        
        return mock_frame

    def _initialize_models(self):
        """Initialize required models"""
        try:
            from ultralytics import YOLO
            
            # Initialize models if path exists
            if settings.FACE_MODEL_PATH.exists():
                self.face_model = YOLO(str(settings.FACE_MODEL_PATH))
                logger.info("Face detection model loaded successfully")
            
            if settings.POSE_MODEL_PATH.exists():
                self.pose_model = YOLO(str(settings.POSE_MODEL_PATH))
                logger.info("Pose detection model loaded successfully")
            
            if settings.OBJECT_MODEL_PATH.exists():
                self.object_model = YOLO(str(settings.OBJECT_MODEL_PATH))
                logger.info("Object detection model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            logger.info("Continuing with mock data only")

    async def process_frame(self, frame: np.ndarray, detection_type: str = "all") -> List[Dict]:
        """Process a single frame and return detections"""
        detections = []
        
        try:
            # Select appropriate device
            device = 0 if torch.cuda.is_available() else 'cpu'
            
            # Convert frame to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if detection_type in ["all", "face"] and self.face_model:
                face_results = self.face_model(frame_rgb, conf=0.5, device=device)
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
            
            if detection_type in ["all", "object"] and self.object_model:
                object_results = self.object_model(frame_rgb, conf=0.5, device=device)
                for result in object_results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        class_name = self.object_model.names[int(cls_id)]
                        detections.append({
                            "type": "object",
                            "confidence": float(conf),
                            "bbox": box.tolist(),
                            "class_name": class_name
                        })
            
            if detection_type in ["all", "pose"] and self.pose_model:
                pose_results = self.pose_model(frame_rgb, conf=0.5, device=device)
                for result in pose_results:
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        for i in range(len(boxes)):
                            bbox = boxes[i].tolist()
                            conf = float(confidences[i])
                            kpts = keypoints[i].tolist() if i < len(keypoints) else []
                            
                            detections.append({
                                "type": "pose",
                                "confidence": conf,
                                "bbox": bbox,
                                "class_name": "person",
                                "keypoints": kpts
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return []

    async def process_video(self, video_path: str, video_id: int, detection_type: str) -> Dict:
        """
        Process a video file for various detection types with real-time updates.
        
        Args:
            video_path (str): Path to the input video file
            video_id (int): Unique identifier for the video
            detection_type (str): Type of detection to perform (e.g., 'theft', 'loitering', 'face_detection')
        
        Returns:
            Dict: Comprehensive processing results including detections, output paths, and metadata
        """
        try:
            # Ensure necessary directories exist
            os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
            os.makedirs(settings.THUMBNAILS_DIR, exist_ok=True)
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Extract video metadata
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Prepare output video writer
            output_path = settings.PROCESSED_DIR / f"processed_{video_id}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Initialize processing variables
            frame_number = 0
            detections = []
            thumbnail_saved = False
            last_progress_update = 0
            
            # Process video frames
            while cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame to improve performance
                if frame_number % settings.SKIP_FRAMES == 0:
                    # Convert frame to RGB for AI models
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    try:
                        # Perform detection based on detection type
                        frame_detections = await self._detect_in_frame(
                            frame_rgb, 
                            detection_type
                        )
                        
                        # Add frame number to detections
                        for det in frame_detections:
                            det["frame_number"] = frame_number
                            detections.append(det)
                        
                        # Draw detection boxes on frame
                        for det in frame_detections:
                            x1, y1, x2, y2 = map(int, det["bbox"])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{det.get('class_name', 'Object')} {det.get('confidence', 0):.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save thumbnail for the first processed frame
                        if not thumbnail_saved:
                            thumbnail_path = settings.THUMBNAILS_DIR / f"thumb_{video_id}.jpg"
                            cv2.imwrite(str(thumbnail_path), frame)
                            thumbnail_saved = True
                        
                        # Broadcast detections via WebSocket
                        for detection in frame_detections:
                            await manager.broadcast(json.dumps({
                                "type": "detection",
                                "video_id": video_id,
                                "detection": detection
                            }))
                    
                    except Exception as detection_error:
                        logger.error(f"Detection error: {str(detection_error)}")
                
                # Write processed frame to output video
                out.write(frame)
                frame_number += 1
                
                # Periodic progress updates via WebSocket
                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    
                    # Avoid too frequent updates
                    if progress - last_progress_update >= 5:
                        await manager.broadcast(json.dumps({
                            "type": "processing_progress",
                            "video_id": video_id,
                            "progress": progress,
                            "total_frames": total_frames
                        }))
                        last_progress_update = progress
            
            # Release video resources
            cap.release()
            out.release()
            
            # Ensure thumbnail is saved if not already done
            if not thumbnail_saved and frame is not None:
                thumbnail_path = settings.THUMBNAILS_DIR / f"thumb_{video_id}.jpg"
                cv2.imwrite(str(thumbnail_path), frame)
            
            # Prepare and return processing results
            processing_results = {
                "video_id": video_id,
                "output_path": str(output_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_saved else None,
                "detections": detections,
                "total_frames": frame_number,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "detection_type": detection_type
            }
            
            # Final broadcast of processing completion
            await manager.broadcast(json.dumps({
                "type": "processing_completed",
                "video_id": video_id,
                "results": processing_results
            }))
            
            return processing_results
        
        except Exception as e:
            # Handle and log any unexpected errors
            logger.error(f"Error processing video {video_id}: {str(e)}")
            
            # Broadcast error message
            await manager.broadcast(json.dumps({
                "type": "processing_error",
                "video_id": video_id,
                "error": str(e)
            }))
            
            raise

    async def process_theft_detection_frame(self, frame):
        """
        Process a frame through the theft detection model
        
        Args:
            frame: Video frame to process
            
        Returns:
            Dict with detection results
        """
        if frame is None:
            return {"detected": False, "confidence": 0, "bounding_boxes": []}
        
        # Process the frame with the theft detection model
        # Use a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.theft_detection_model.detect, frame
        )
        
        return result
    
    async def process_loitering_detection_frame(self, frame, camera_id="default"):
        """
        Process a frame through the loitering detection model
        
        Args:
            frame: Video frame to process
            camera_id: Camera identifier
            
        Returns:
            Dict with detection results
        """
        if frame is None:
            return {"detected": False, "duration": 0, "regions": []}
        
        # Process the frame with the loitering detection model
        # Use a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.loitering_detection_model.detect(frame, camera_id)
        )
        
        return result

class TheftDetectionModel:
    """Simulates theft detection using computer vision"""
    
    def __init__(self):
        logger.info("Theft Detection Model initialized")
        self._last_detections = {}
        
    def detect(self, frame):
        """
        Perform theft detection on the frame
        Returns detection results with bounding boxes and confidence scores
        """
        try:
            if frame is None:
                return {"detected": False, "confidence": 0, "bounding_boxes": []}
                
            # Generate simulated detections
            height, width = frame.shape[:2]
            
            # Use time to create moving detection areas
            t = time.time() % 30  # 30 second cycle
            
            # Probability of detection increases over time then resets
            # More aggressive detection probability for demonstration purposes
            detection_probability = min(0.95, (t % 10) / 6)  # Reaches peak faster
            
            # Increase the chance of detecting something - 50% chance instead of 30%
            detected = random.random() < detection_probability * 0.5
            confidence = random.uniform(0.75, 0.98) if detected else random.uniform(0.2, 0.4)
            
            bounding_boxes = []
            if detected:
                # Create a moving bounding box
                center_x = int(width * (0.3 + 0.4 * math.sin(t * 0.5)))
                center_y = int(height * (0.3 + 0.4 * math.cos(t * 0.5)))
                box_width = int(width * 0.2)
                box_height = int(height * 0.3)
                
                x1 = max(0, center_x - box_width // 2)
                y1 = max(0, center_y - box_height // 2)
                x2 = min(width, x1 + box_width)
                y2 = min(height, y1 + box_height)
                
                bounding_boxes.append([x1, y1, x2, y2])
                
                # Add a second box occasionally for multiple detection demonstration
                if random.random() < 0.3:
                    second_x = int(width * (0.6 + 0.2 * math.cos(t * 0.7)))
                    second_y = int(height * (0.6 + 0.2 * math.sin(t * 0.7)))
                    second_width = int(width * 0.15)
                    second_height = int(height * 0.25)
                    
                    x1_2 = max(0, second_x - second_width // 2)
                    y1_2 = max(0, second_y - second_height // 2)
                    x2_2 = min(width, x1_2 + second_width)
                    y2_2 = min(height, y1_2 + second_height)
                    
                    bounding_boxes.append([x1_2, y1_2, x2_2, y2_2])
                
                # Draw bounding box on the frame for debugging
                for box in bounding_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Theft: {confidence:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            return {
                "detected": detected,
                "confidence": confidence,
                "bounding_boxes": bounding_boxes
            }
            
        except Exception as e:
            logger.error(f"Error in theft detection: {str(e)}")
            return {"detected": False, "confidence": 0, "bounding_boxes": [], "error": str(e)}


class LoiteringDetectionModel:
    """Simulates loitering detection using computer vision"""
    
    def __init__(self):
        logger.info("Loitering Detection Model initialized")
        # Track person locations over time to detect loitering
        self._person_tracks = {}
        self._last_detection_time = {}
        self._durations = {}  # Track durations for each camera to make them increase over time
        
    def detect(self, frame, camera_id="default"):
        """
        Perform loitering detection on the frame
        Returns detection results with loitering regions and duration
        """
        try:
            if frame is None:
                return {"detected": False, "duration": 0, "regions": []}
                
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Use time to create simulated loitering
            current_time = time.time()
            t = current_time % 60  # 60 second cycle
            
            # Track detection for this camera
            if camera_id not in self._last_detection_time:
                self._last_detection_time[camera_id] = current_time
                self._person_tracks[camera_id] = []
                self._durations[camera_id] = 0
            
            # Time in the same area = loitering duration
            time_diff = current_time - self._last_detection_time[camera_id]
            
            # Generate simulated loitering - more likely in certain regions
            x_region = int(width * (0.6 + 0.3 * math.sin(t * 0.1)))
            y_region = int(height * (0.6 + 0.3 * math.cos(t * 0.1)))
            
            # More likely to detect loitering - increased probability
            loitering_probability = min(0.9, time_diff / 20) * 0.6
            
            # Random detection with increasing probability over time
            detected = random.random() < loitering_probability
            
            regions = []
            if detected:
                # If already detected, increase duration for persistence
                if camera_id in self._durations and self._durations[camera_id] > 0:
                    self._durations[camera_id] += random.uniform(0.2, 0.5)  # Gradual increase
                else:
                    # New detection
                    self._durations[camera_id] = random.uniform(3.0, 8.0)  # Start with a higher value
                
                duration = self._durations[camera_id]
                
                # Create a region of interest where loitering is detected
                region_width = int(width * 0.25)
                region_height = int(height * 0.25)
                
                x = max(0, x_region - region_width // 2)
                y = max(0, y_region - region_height // 2)
                
                regions.append({
                    "x": x,
                    "y": y,
                    "width": region_width,
                    "height": region_height
                })
                
                # Add a second region occasionally
                if random.random() < 0.25 and duration > 10:
                    second_x = int(width * (0.3 + 0.2 * math.sin(t * 0.2)))
                    second_y = int(height * (0.3 + 0.2 * math.cos(t * 0.2)))
                    
                    regions.append({
                        "x": max(0, second_x - region_width // 2),
                        "y": max(0, second_y - region_height // 2),
                        "width": region_width,
                        "height": region_height
                    })
                
                # Draw regions on the frame for debugging
                for region in regions:
                    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Loitering: {duration:.1f}s", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # Reset tracking with a small probability to avoid getting stuck in detected state
                if random.random() < 0.1:
                    self._durations[camera_id] = 0
                    self._last_detection_time[camera_id] = current_time
                # Otherwise keep a minimum duration to avoid flickering
                elif camera_id in self._durations and self._durations[camera_id] > 0:
                    duration = max(1.0, self._durations[camera_id] - 0.5)  # Slow decrease
                    self._durations[camera_id] = duration
                else:
                    duration = 0
            
            return {
                "detected": detected,
                "duration": self._durations.get(camera_id, 0),
                "regions": regions
            }
            
        except Exception as e:
            logger.error(f"Error in loitering detection: {str(e)}")
            return {"detected": False, "duration": 0, "regions": [], "error": str(e)}

video_processor = VideoProcessor()