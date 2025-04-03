import os
# Force torch to not load weights only
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
# Enable GPU memory growth to avoid allocating all memory at once
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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
import torch
from database import update_customer_face_encoding, add_detection, add_incident
import threading
import nest_asyncio
import json

logger = logging.getLogger(__name__)

# Check for CUDA availability and log it
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
    logger.info(f"CUDA is available: {device_count} device(s). Main device: {device_name}")
    # Set default device to 0 (first GPU)
    DEVICE = 0
else:
    logger.warning("CUDA is not available. Using CPU.")
    DEVICE = 'cpu'

# Configure PyTorch for improved performance
if CUDA_AVAILABLE:
    # Enable TF32 precision (on Ampere GPUs and later)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cudnn benchmark mode for faster training
    torch.backends.cudnn.benchmark = True

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
        
        # Frame batch processing for GPU
        self.frame_batch = []
        self.frame_batch_info = []
        self.max_batch_size = 8 if CUDA_AVAILABLE else 1
        
        # Resolution settings for faster processing - 640p instead of 720p
        self.process_width = 640  
        self.process_height = 480
        self.resize_frames = True
        
        # Display mode settings
        self.display_mode = "raw"  # Options: "raw" (no inference visualization), "detection" (with visualization)
        self.run_detection_in_background = True  # Run detection but don't show visualization
        
        # Initialize inference models
        self.theft_detection_model = TheftDetectionModel()
        self.loitering_detection_model = LoiteringDetectionModel()
        
        logger.info(f"VideoProcessor initialized with batch size: {self.max_batch_size}, processing resolution: {self.process_width}x{self.process_height}, display mode: {self.display_mode}")
        
    def resize_frame(self, frame):
        """Resize frame to 720p for faster processing"""
        if frame is None or not self.resize_frames:
            return frame
            
        h, w = frame.shape[:2]
        
        # Skip if already smaller than target resolution
        if w <= self.process_width and h <= self.process_height:
            return frame
            
        # Calculate aspect ratio preserving dimensions
        ratio = min(self.process_width / w, self.process_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # Resize the frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized
        
    def configure_resolution(self, enable_resize=True, width=1280, height=720):
        """Configure frame resizing for processing"""
        self.resize_frames = enable_resize
        if width > 0 and height > 0:
            self.process_width = width
            self.process_height = height
        
        logger.info(f"Video processor resolution settings updated: resize={self.resize_frames}, target={self.process_width}x{self.process_height}")
        return {
            "resize_enabled": self.resize_frames,
            "width": self.process_width, 
            "height": self.process_height
        }
        
    def configure_display_mode(self, mode="raw", run_detection=True):
        """
        Configure how video is displayed and processed
        Args:
            mode: "raw" (no visualization) or "detection" (with visualization)
            run_detection: Whether to run detection processes in background
        Returns:
            Current display settings
        """
        self.display_mode = mode
        self.run_detection_in_background = run_detection
        
        logger.info(f"Video processor display mode updated: mode={self.display_mode}, background_detection={self.run_detection_in_background}")
        return {
            "display_mode": self.display_mode,
            "background_detection": self.run_detection_in_background
        }

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
                
                # Resize the frame to 720p for faster processing
                frame = self.resize_frame(frame)
        
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
                            
                            # Skip if not enough keypoints for full analysis
                            if len(kpts) < 17:
                                continue
                            
                            # Extract features for tracking
                            features = self.extract_person_features(frame, bbox)
                            
                            # Match with existing tracked persons
                            matched_id = self.match_person(features, bbox, self.frame_count)
                            
                            if matched_id is not None:
                                # Update existing person
                                person_id = matched_id
                                person_data = self.tracked_persons[person_id]
                                
                                # Update tracking info
                                person_data.update({
                                    'bbox': bbox,
                                    'keypoints': kpts,
                                    'features': features,
                                    'last_seen_frame': self.frame_count,
                                    'last_seen_time': current_time
                                })
                                
                                # Calculate time in frame
                                time_in_frame = current_time - person_data['first_seen_time']
                                
                                # Track accumulated time
                                if 'accumulated_time' not in person_data:
                                    person_data['accumulated_time'] = 0
                                
                                # Add time spent since last frame
                                person_data['accumulated_time'] += time_diff
                                
                                accumulated_time = person_data['accumulated_time']
                                
                                # Add to current detections
                                current_detections.append({
                                    'id': person_id,
                                    'bbox': bbox,
                                    'time_in_frame': time_in_frame,
                                    'accumulated_time': accumulated_time,
                                    'confidence': float(conf)
                                })
                            else:
                                # New person detected
                                person_id = f"person_{self.next_person_id}"
                                self.next_person_id += 1
                                
                                self.tracked_persons[person_id] = {
                                    'bbox': bbox,
                                    'keypoints': kpts,
                                    'features': features,
                                    'first_seen_frame': self.frame_count,
                                    'first_seen_time': current_time,
                                    'last_seen_frame': self.frame_count,
                                    'last_seen_time': current_time,
                                    'accumulated_time': 0
                                }
                                
                                # Add to current detections
                                current_detections.append({
                                    'id': person_id,
                                    'bbox': bbox,
                                    'time_in_frame': 0,
                                    'accumulated_time': 0,
                                    'confidence': float(conf)
                                })
                
                # Update last processed detections
                self.last_processed_detections = current_detections
            else:
                # Use previously processed detections
                current_detections = self.last_processed_detections
            
            # Process all current detections for visualization and theft detection
            for detection in current_detections:
                person_id = detection['id']
                bbox = detection['bbox']
                keypoints = detection['keypoints']
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, bbox)
                width, height = x2 - x1, y2 - y1
                
                # Define detection zones
                chest_box = [x1 + int(0.1 * width), y1, x2 - int(0.1 * width), y1 + int(0.4 * height)]
                left_waist_box = [x1, y1 + int(0.5 * height), x1 + int(0.5 * width), y2]
                right_waist_box = [x1 + int(0.5 * width), y1 + int(0.5 * height), x2, y2]
                
                # Draw detection zones
                cv2.rectangle(frame, tuple(chest_box[:2]), tuple(chest_box[2:]), (0, 255, 255), 2)
                cv2.rectangle(frame, tuple(left_waist_box[:2]), tuple(left_waist_box[2:]), (255, 0, 0), 2)
                cv2.rectangle(frame, tuple(right_waist_box[:2]), tuple(right_waist_box[2:]), (0, 0, 255), 2)
                
                # Draw skeleton
                self.draw_skeleton(frame, keypoints)
                
                # Get wrist and nose positions
                left_wrist = keypoints[self.LEFT_WRIST] if self.LEFT_WRIST < len(keypoints) and keypoints[self.LEFT_WRIST][0] > 0 else None
                right_wrist = keypoints[self.RIGHT_WRIST] if self.RIGHT_WRIST < len(keypoints) and keypoints[self.RIGHT_WRIST][0] > 0 else None
                nose = keypoints[self.NOSE] if self.NOSE < len(keypoints) and keypoints[self.NOSE][0] > 0 else None
                
                # Approximate eye gaze
                is_looking_at_chest = self.approximate_eye_gaze(nose, chest_box)
                
                # Track hands in chest/waist regions
                for wrist, hand_label in [(left_wrist, f"{person_id}_left"), (right_wrist, f"{person_id}_right")]:
                    if wrist is None:
                        continue
                    
                    wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                    wrist_box = [wrist_x - 10, wrist_y - 10, wrist_x + 10, wrist_y + 10]
                    cv2.rectangle(frame, (wrist_box[0], wrist_box[1]), (wrist_box[2], wrist_box[3]), (0, 255, 0), 2)
                    
                    # Check if hand is in a suspicious zone
                    in_chest = self.is_intersecting(wrist_box, chest_box)
                    in_left_waist = self.is_intersecting(wrist_box, left_waist_box)
                    in_right_waist = self.is_intersecting(wrist_box, right_waist_box)
                    
                    # Initialize tracking for this hand if not already tracked
                    if hand_label not in self.hand_positions:
                        self.hand_positions[hand_label] = []
                    
                    # Record current hand position and time
                    if in_chest:
                        zone = "chest"
                    elif in_left_waist:
                        zone = "left_waist"
                    elif in_right_waist:
                        zone = "right_waist"
                    else:
                        zone = "other"
                    
                    # Add to hand position history
                    self.hand_positions[hand_label].append((current_time, (wrist_x, wrist_y), zone))
                    
                    # Limit history size to avoid memory issues
                    if len(self.hand_positions[hand_label]) > 60:  # Assuming 30fps, keep 2 seconds of history
                        self.hand_positions[hand_label].pop(0)
                    
                    # Analyze hand positions for theft detection
                    if in_chest or in_left_waist or in_right_waist:
                        # Determine which zone and appropriate threshold
                        zone_type = "chest" if in_chest else "waist"
                        threshold = hand_stay_time_chest if in_chest else hand_stay_time_waist
                        
                        # Count how long the hand has been in this zone continuously
                        zone_duration = 0
                        continuous_in_zone = True
                        
                        # Check hand position history in reverse order
                        for idx in range(len(self.hand_positions[hand_label]) - 1, 0, -1):
                            past_time, _, past_zone = self.hand_positions[hand_label][idx]
                            prev_time, _, prev_zone = self.hand_positions[hand_label][idx - 1]
                            
                            # If zone changes, break the continuity
                            if past_zone != zone and past_zone != "other":
                                continuous_in_zone = False
                                break
                            
                            # If zone matches, add to duration
                            if past_zone == zone:
                                zone_duration += (past_time - prev_time)
                        
                        # Add suspicious region for visualization
                        suspicious_regions.append({
                            "x": wrist_x - 10,
                            "y": wrist_y - 10,
                            "width": 20,
                            "height": 20,
                            "zone": zone_type,
                            "duration": zone_duration
                        })
                        
                        # Check if duration exceeds threshold
                        if zone_duration >= threshold:
                            # Generate a unique incident ID combining person and zone
                            incident_id = f"{person_id}_{zone}"
                            
                            # Check if this incident has already been detected
                            if incident_id not in self.detected_theft_incidents:
                                # Mark as detected
                                self.detected_theft_incidents.add(incident_id)
                                
                                # Log detection
                                logger.info(f"Theft detection: Person {person_id}, hand in {zone} for {zone_duration:.1f}s, looking: {is_looking_at_chest}")
                                
                                # Save incident screenshot
                                self.save_theft_incident(
                                    person_id, frame, bbox, keypoints, zone_type, is_looking_at_chest
                                )
                                
                                # Add to incidents list
                                theft_incidents.append({
                                    "type": "theft",
                                    "person_id": person_id,
                                    "confidence": 0.8,
                                    "bbox": bbox,
                                    "zone": zone_type,
                                    "looking_at_hands": is_looking_at_chest,
                                    "duration": zone_duration
                                })
                
                # Visualization - red box for detected theft, green otherwise
                incident_id_chest = f"{person_id}_chest"
                incident_id_waist = f"{person_id}_waist"
                
                if incident_id_chest in self.detected_theft_incidents or incident_id_waist in self.detected_theft_incidents:
                    # Detected theft - red box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"ID: {person_id} - THEFT", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Normal tracking - green box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Clean up old tracked persons to prevent memory issues (every 100 frames)
            if self.frame_count % 100 == 0:
                current_ids = set(detection['id'] for detection in current_detections)
                for person_id in list(self.tracked_persons.keys()):
                    # Remove if not seen in the last 60 frames (2 seconds at 30fps)
                    if person_id not in current_ids and \
                    'last_seen_frame' in self.tracked_persons[person_id] and \
                    self.frame_count - self.tracked_persons[person_id]['last_seen_frame'] > 60:
                        del self.tracked_persons[person_id]
                
                # Also clean up hand positions for persons no longer tracked
                valid_persons = set(self.tracked_persons.keys())
                for hand_label in list(self.hand_positions.keys()):
                    person_id = hand_label.split('_')[0]
                    if person_id not in valid_persons:
                        del self.hand_positions[hand_label]
            
            return {
                "video_id": video_id,
                "output_path": str(output_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_saved else None,
                "detections": detections,
                "total_frames": frame_number,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "detection_type": detection_type
            }
            
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

    async def process_theft_detection(self, video_path: str, output_path: str, screenshots_dir: str, hand_stay_time: float = 2.0, camera_id: Optional[int] = None) -> Dict:
        """
        Process video for theft detection with improved tracking
        Args:
            video_path: Path to input video file
            output_path: Path to save processed video
            screenshots_dir: Directory to save screenshots
            hand_stay_time: Time threshold for hand position
            camera_id: Optional camera ID for logging
        Returns:
            Dictionary with processing results
        """
        return await self.process_theft_detection_smooth(
            video_path=video_path,
            output_path=output_path,
            screenshots_dir=screenshots_dir,
            hand_stay_time=hand_stay_time,
            camera_id=camera_id
        )
    
    async def process_loitering_detection(self, video_path: str, output_path: str, camera_id: Optional[int] = None, threshold_time: float = 10) -> Dict:
        """
        Process video for loitering detection with improved tracking
        Args:
            video_path: Path to input video file
            output_path: Path to save processed video
            camera_id: Optional camera ID for logging
            threshold_time: Time threshold (seconds) to consider loitering
        Returns:
            Dictionary with processing results
        """
        return await self.process_loitering_detection_smooth(
            video_path=video_path,
            output_path=output_path,
            camera_id=camera_id,
            threshold_time=threshold_time
        )

    def add_frame_to_batch(self, frame, camera_id="default"):
        """Add a frame to the processing batch"""
        if frame is None:
            return False
        
        # Resize the frame to 720p for faster processing
        resized_frame = self.resize_frame(frame)
            
        self.frame_batch.append(resized_frame)
        self.frame_batch_info.append({"camera_id": camera_id})
        
        # Return True if batch is full
        return len(self.frame_batch) >= self.max_batch_size
        
    async def process_batch(self, detection_type="all"):
        """Process a batch of frames for detection"""
        if not self.frame_batch:
            return []
            
        results = []
        
        try:
            # Skip detection completely if display mode is "raw" and background detection is disabled
            if self.display_mode == "raw" and not self.run_detection_in_background:
                # Return empty results, no detection performed
                return [{"type": "no_detection", "result": None} for _ in self.frame_batch]
            
            # For theft detection, we need to process frames sequentially
            # since tracking information is maintained across frames
            for i, frame in enumerate(self.frame_batch):
                info = self.frame_batch_info[i]
                camera_id = info.get("camera_id", "default")
                
                if detection_type in ["all", "theft"]:
                    # Process for theft detection
                    # Create a copy if we're in raw mode to avoid modifying the original frame
                    detection_frame = frame.copy() if self.display_mode == "raw" else frame
                    
                    # Process the frame
                    theft_result = self.theft_detection_model.detect(
                        frame=detection_frame, 
                        camera_id=camera_id,
                        draw_visualization=(self.display_mode == "detection")
                    )
                    results.append({"type": "theft", "result": theft_result})
                
                if detection_type in ["all", "loitering"]:
                    # Process for loitering detection
                    # Create a copy if we're in raw mode to avoid modifying the original frame
                    detection_frame = frame.copy() if self.display_mode == "raw" else frame
                    
                    # Process the frame
                    loitering_result = self.loitering_detection_model.detect(
                        frame=detection_frame, 
                        camera_id=camera_id,
                        draw_visualization=(self.display_mode == "detection")
                    )
                    
                    if detection_type == "all":
                        # If "all", append to existing result
                        results[i]["loitering"] = loitering_result
                    else:
                        # Otherwise create new result
                        results.append({"type": "loitering", "result": loitering_result})
        except Exception as e:
            logger.error(f"Error processing frame batch: {str(e)}")
            
        # Clear the batch
        self.frame_batch = []
        self.frame_batch_info = []
        
        return results

class TheftDetectionModel:
    """Theft detection using pose estimation with persistent tracking"""
    
    def __init__(self):
        logger.info("Theft Detection Model initialized")
        # Initialize pose detection model
        self.pose_model = None
        try:
            from ultralytics import YOLO
            # Use small model with half precision for RTX 4090
            model_path = "yolov8n-pose.pt"
            self.pose_model = YOLO(model_path)
            
            # Use FP16 precision on RTX 4090
            if CUDA_AVAILABLE:
                logger.info(f"Using GPU acceleration for pose model on {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Using CPU for pose model")
        except Exception as e:
            logger.error(f"Failed to load pose detection model: {str(e)}")
        
        # Define keypoint indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        
        # Initialize tracking state
        self.tracked_persons = {}  # {person_id: {last_seen, bbox, keypoints, features, etc.}}
        self.hand_positions = {}   # {person_id_hand: [(time, position, zone),...]}
        self.detected_theft_incidents = set()  # Track already detected incidents
        self.next_person_id = 1
        
        # Screenshot directory for incidents
        self.screenshots_dir = os.path.join(settings.SCREENSHOTS_DIR, f"theft_{int(time.time())}")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Detection parameters
        self.hand_stay_time_chest = settings.HAND_STAY_TIME_CHEST  # Default 2.0 seconds
        self.hand_stay_time_waist = settings.HAND_STAY_TIME_WAIST  # Default 1.5 seconds
        self.crop_padding = settings.CROP_PADDING  # Default padding for cropped screenshots
        
        # Processing state
        self.device = DEVICE  # Use global DEVICE constant (GPU if available)
        self.frame_count = 0
        self.last_frame_time = None
        self.last_processed_detections = []
        
        # Optimization: Set the processing interval (every Nth frame)
        self.process_interval = 6  # Process every 6th frame with 640p (more aggressive skipping)
        
        # Optimization: Setup database thread
        self.db_worker_running = False
        self.db_queue = []
        self._start_db_worker()
    
    def _start_db_worker(self):
        """Start a dedicated thread for database operations"""
        if not self.db_worker_running:
            self.db_worker_running = True
            threading.Thread(target=self._db_worker, daemon=True).start()
            logger.info("Started database worker thread for theft detection")
    
    def _db_worker(self):
        """Background thread that processes database operations in batch"""
        import time
        import asyncio
        
        # Create a dedicated event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self.db_worker_running:
                # Process items if we have enough in queue or after waiting
                if len(self.db_queue) > 0:
                    # Take a copy of current queue items and clear the queue
                    items_to_process = self.db_queue.copy()
                    self.db_queue = []
                    
                    # Process all items in a single async batch
                    loop.run_until_complete(self._process_db_batch(items_to_process))
                
                # Sleep to avoid busy waiting
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error in theft detection DB worker: {str(e)}")
        finally:
            loop.close()
            self.db_worker_running = False
    
    async def _process_db_batch(self, items):
        """Process a batch of database operations"""
        try:
            # Import database functions
            from database import add_detection, add_incident
            
            # Process each item
            for item in items:
                person_id, bbox, zone_type, behavior_desc, image_path, frame_number = item
                
                # Add detection
                detection_data = {
                    "camera_id": None,
                    "timestamp": datetime.now(),
                    "frame_number": frame_number,
                    "detection_type": "theft",
                    "confidence": 0.8,
                    "bbox": bbox,
                    "class_name": "person",
                    "image_path": image_path,
                    "detection_metadata": {
                        "person_id": person_id,
                        "zone_type": zone_type,
                        "behavior": behavior_desc
                    }
                }
                
                # Add detection first
                detection_id = await add_detection(detection_data)
                if detection_id:
                    # Create incident with the detection ID
                    incident_data = {
                        "type": "theft",
                        "timestamp": datetime.now(),
                        "location": "Unknown",
                        "description": f"Suspicious hand movement in {zone_type} area while {behavior_desc}",
                        "image_path": image_path,
                        "video_url": None,
                        "severity": "high",
                        "confidence": 0.8,
                        "detection_ids": [detection_id],
                        "is_resolved": False,
                        "frame_number": frame_number
                    }
                    
                    # Add incident record
                    incident_id = await add_incident(incident_data)
                    logger.info(f"Created theft incident with ID {incident_id} for person {person_id}")
                else:
                    logger.error(f"Failed to create detection record for person {person_id}")
        except Exception as e:
            logger.error(f"Error processing theft detection database batch: {str(e)}")

    def save_theft_incident(self, person_id, frame, bbox, keypoints, zone_type, looking_at_chest=False):
        """Save theft incident screenshot synchronously"""
        try:
            # Create a filename for the incident image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            incident_filename = f"theft_{person_id}_{timestamp}.jpg"
            incident_image_path = os.path.join(self.screenshots_dir, incident_filename)
            
            # Create a copy of the frame for annotation
            annotated_frame = frame.copy()
            
            # Draw bounding box and skeleton on the annotated frame
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            self.draw_skeleton(annotated_frame, keypoints, color=(255, 0, 0))
            
            # Add text label
            behavior_desc = "not looking at hands" if not looking_at_chest else "looking at hands"
            cv2.putText(annotated_frame, 
                    f"THEFT: Hand in {zone_type}, {behavior_desc}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Crop the suspect with padding
            x1_crop = max(0, x1 - self.crop_padding)
            y1_crop = max(0, y1 - self.crop_padding)
            x2_crop = min(frame.shape[1], x2 + self.crop_padding)
            y2_crop = min(frame.shape[0], y2 + self.crop_padding)
            suspect_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            # Save both the annotated full frame and the cropped suspect
            cv2.imwrite(incident_image_path, annotated_frame)
            crop_filename = f"crop_{person_id}_{timestamp}.jpg"
            crop_path = os.path.join(self.screenshots_dir, crop_filename)
            cv2.imwrite(crop_path, suspect_crop)
            
            logger.info(f"Saved theft incident images for person {person_id} in {zone_type}")
            
            # Add to queue for database operations instead of starting a new thread
            self.db_queue.append((person_id, bbox, zone_type, behavior_desc, crop_path, self.frame_count))
            
            return incident_image_path
            
        except Exception as e:
            logger.error(f"Error saving theft incident: {str(e)}")
            return None
    
    def _add_to_database(self, person_id, bbox, zone_type, behavior_desc, image_path, frame_number):
        """Legacy method kept for compatibility - now adds to queue instead"""
        self.db_queue.append((person_id, bbox, zone_type, behavior_desc, image_path, frame_number))
        return True

    def detect(self, frame, camera_id="default", hand_stay_time=None, draw_visualization=True):
        """
        Perform theft detection on a single frame
        Args:
            frame: Input video frame
            camera_id: Camera identifier 
            hand_stay_time: Custom time threshold for detection
            draw_visualization: Whether to draw visualization on the frame
        Returns:
            Detection results with theft incidents
        """
        try:
            if frame is None or self.pose_model is None:
                return {"detected": False, "incidents": [], "regions": []}
            
            # Use default hand stay time if not provided
            if hand_stay_time is None:
                hand_stay_time_chest = self.hand_stay_time_chest
                hand_stay_time_waist = self.hand_stay_time_waist
            else:
                hand_stay_time_chest = hand_stay_time
                hand_stay_time_waist = hand_stay_time * 0.75  # Waist has lower threshold
            
            # Initialize time tracking for first frame
            current_time = time.time()
            if self.last_frame_time is None:
                self.last_frame_time = current_time
                
            # Calculate time between frames
            time_diff = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Increment frame counter
            self.frame_count += 1
            
            # Process every Nth frame for efficiency
            current_detections = []
            theft_incidents = []
            suspicious_regions = []
            
            # Only perform detection on specific frames for efficiency
            should_process = self.frame_count % self.process_interval == 0
            
            if should_process:
                # Convert frame to RGB for pose detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Optimize inference parameters for 640p
                conf_threshold = 0.35  # Lower confidence threshold for faster processing with 640p
                iou_threshold = 0.5  # Higher IoU threshold to filter duplicates
                
                # Use half precision with RTX 4090 for better performance
                if CUDA_AVAILABLE and self.device == 0:
                    with torch.cuda.amp.autocast():
                        pose_results = self.pose_model(
                            frame_rgb, 
                            conf=conf_threshold,
                            iou=iou_threshold,
                            device=self.device,
                            verbose=False
                        )
                else:
                    pose_results = self.pose_model(
                        frame_rgb, 
                        conf=conf_threshold,
                        iou=iou_threshold,
                        device=self.device,
                        verbose=False
                    )
                
                # Process each detection and maintain consistent IDs
                for result in pose_results:
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        for i in range(len(boxes)):
                            bbox = boxes[i].tolist()
                            kpts = keypoints[i].tolist() if i < len(keypoints) else []
                            
                            # Skip if not enough keypoints for full analysis
                            if len(kpts) < 17:
                                continue
                            
                            # Extract features for tracking
                            features = self.extract_person_features(frame, bbox)
                            
                            # Match with existing tracked persons
                            matched_id = self.match_person(features, bbox, self.frame_count)
                            
                            if matched_id is not None:
                                # Update existing person
                                person_id = matched_id
                                person_data = self.tracked_persons[person_id]
                                
                                # Update tracking info
                                person_data.update({
                                    'bbox': bbox,
                                    'keypoints': kpts,
                                    'features': features,
                                    'last_seen_frame': self.frame_count,
                                    'last_seen_time': current_time
                                })
                                
                                # Calculate time in frame
                                time_in_frame = current_time - person_data['first_seen_time']
                                
                                # Track accumulated time
                                if 'accumulated_time' not in person_data:
                                    person_data['accumulated_time'] = 0
                                
                                # Add time spent since last frame
                                person_data['accumulated_time'] += time_diff
                                
                                accumulated_time = person_data['accumulated_time']
                                
                                # Add to current detections
                                current_detections.append({
                                    'id': person_id,
                                    'bbox': bbox,
                                    'time_in_frame': time_in_frame,
                                    'accumulated_time': accumulated_time,
                                    'confidence': float(confidences[i])
                                })
                                
                                # Check if person has been loitering
                                if accumulated_time > hand_stay_time_chest:
                                    # Check if this loitering has already been detected
                                    if person_id not in self.detected_theft_incidents:
                                        # Save incident screenshot
                                        image_path = self.save_theft_incident(
                                            person_id, frame, bbox, keypoints, "chest", False
                                        )
                                        
                                        # Add to incidents list
                                        theft_incidents.append({
                                            "type": "theft",
                                            "person_id": person_id,
                                            "confidence": float(confidences[i]),
                                            "bbox": bbox,
                                            "duration": accumulated_time,
                                            "image_path": image_path
                                        })
                            else:
                                # New person detected
                                person_id = f"person_{self.next_person_id}"
                                self.next_person_id += 1
                                
                                self.tracked_persons[person_id] = {
                                    'bbox': bbox,
                                    'keypoints': kpts,
                                    'features': features,
                                    'first_seen_frame': self.frame_count,
                                    'first_seen_time': current_time,
                                    'last_seen_frame': self.frame_count,
                                    'last_seen_time': current_time,
                                    'accumulated_time': 0
                                }
                                
                                # Add to current detections
                                current_detections.append({
                                    'id': person_id,
                                    'bbox': bbox,
                                    'time_in_frame': 0,
                                    'accumulated_time': 0,
                                    'confidence': float(confidences[i])
                                })
            else:
                # For non-processing frames, just update time in tracked persons
                for person_id, person_data in self.tracked_persons.items():
                    if 'active' in person_data and person_data['active']:
                        if 'accumulated_time' in person_data:
                            person_data['accumulated_time'] += time_diff
            
            # Draw visualization on frame only if requested
            if draw_visualization:
                for person_id, person_data in self.tracked_persons.items():
                    if self.frame_count - person_data.get('last_seen_frame', 0) > 30:
                        continue
                        
                    bbox = person_data.get('bbox')
                    if not bbox:
                        continue
                        
                    accumulated_time = person_data.get('accumulated_time', 0)
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Determine color based on time (green->yellow->red as time increases)
                    if accumulated_time > hand_stay_time_chest:
                        # Red for loitering
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {person_id} - THEFT", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Time: {accumulated_time:.1f}s", (x1, y1 - 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        # Calculate color from green to yellow based on time
                        ratio = min(1.0, accumulated_time / hand_stay_time_chest)
                        g = int(255 * (1 - ratio))
                        r = int(255 * ratio)
                        color = (0, g, r)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Time: {accumulated_time:.1f}s", (x1, y1 - 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Clean up old tracked persons less frequently
            if self.frame_count % 300 == 0:  # Much less frequent cleanup
                current_time = time.time()
                for person_id in list(self.tracked_persons.keys()):
                    last_seen_time = self.tracked_persons[person_id].get('last_seen_time', 0)
                    # Remove if not seen in the last 10 seconds 
                    if current_time - last_seen_time > 10:
                        del self.tracked_persons[person_id]
            
            return {
                "detected": len(theft_incidents) > 0,
                "duration": max([person_data.get('accumulated_time', 0) for person_id, person_data in self.tracked_persons.items()]) if self.tracked_persons else 0,
                "regions": current_detections,
                "incidents": theft_incidents,
                "tracked_persons": len(self.tracked_persons)
            }
            
        except Exception as e:
            logger.error(f"Error in theft detection: {str(e)}")
            return {"detected": False, "duration": 0, "regions": [], "error": str(e)}

class LoiteringDetectionModel:
    """Loitering detection using computer vision with persistent tracking"""
    
    def __init__(self):
        logger.info("Loitering Detection Model initialized")
        # Initialize object detection model
        self.object_model = None
        self._load_model()
        
        # Initialize tracking state
        self.tracked_persons = {}  # {person_id: {first_seen, last_seen, position, features, accumulated_time, etc.}}
        self.next_person_id = 1
        self.loitering_incidents = set()  # For avoiding duplicate detections
        self.screenshots_dir = os.path.join(settings.SCREENSHOTS_DIR, f"loitering_{int(time.time())}")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        self.device = DEVICE  # Use global DEVICE constant (GPU if available)
        
        # Time tracking
        self.last_frame_time = None
        self.frame_count = 0
        
        # Optimization: Set the processing interval (every Nth frame)
        self.process_interval = 12  # Process every 12th frame for loitering with 640p resolution
        
        # Optimization: Setup database thread
        self.db_worker_running = False
        self.db_queue = []
        self._start_db_worker()
    
    def _start_db_worker(self):
        """Start a dedicated thread for database operations"""
        if not self.db_worker_running:
            self.db_worker_running = True
            threading.Thread(target=self._db_worker, daemon=True).start()
            logger.info("Started database worker thread for loitering detection")
    
    def _db_worker(self):
        """Background thread that processes database operations in batch"""
        import time
        import asyncio
        
        # Create a dedicated event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self.db_worker_running:
                # Process items if we have enough in queue or after waiting
                if len(self.db_queue) > 0:
                    # Take a copy of current queue items and clear the queue
                    items_to_process = self.db_queue.copy()
                    self.db_queue = []
                    
                    # Process all items in a single async batch
                    loop.run_until_complete(self._process_db_batch(items_to_process))
                
                # Sleep to avoid busy waiting
                time.sleep(1.0)  # Loitering can wait longer between batches
        except Exception as e:
            logger.error(f"Error in loitering detection DB worker: {str(e)}")
        finally:
            loop.close()
            self.db_worker_running = False
    
    async def _process_db_batch(self, items):
        """Process a batch of database operations"""
        try:
            # Import database functions
            from database import add_detection, add_incident
            
            # Process each item
            for item in items:
                person_id, bbox, accumulated_time, camera_id, image_path, frame_number = item
                
                # Add detection
                detection_data = {
                    "camera_id": camera_id,
                    "timestamp": datetime.now(),
                    "frame_number": frame_number,
                    "detection_type": "loitering",
                    "confidence": 0.8,
                    "bbox": bbox,
                    "class_name": "person",
                    "image_path": image_path,
                    "detection_metadata": {
                        "person_id": person_id,
                        "accumulated_time": accumulated_time
                    }
                }
                
                # Add detection first
                detection_id = await add_detection(detection_data)
                if detection_id:
                    # Create incident with the detection ID
                    incident_data = {
                        "type": "loitering",
                        "timestamp": datetime.now(),
                        "location": "Unknown",
                        "description": f"Person loitering for {accumulated_time:.1f} seconds",
                        "image_path": image_path,
                        "video_url": None,
                        "severity": "medium",
                        "confidence": 0.8,
                        "detection_ids": [detection_id],
                        "is_resolved": False,
                        "frame_number": frame_number
                    }
                    
                    # Add incident record
                    incident_id = await add_incident(incident_data)
                    logger.info(f"Created loitering incident with ID {incident_id} for person {person_id}")
                else:
                    logger.error(f"Failed to create detection record for person {person_id}")
        except Exception as e:
            logger.error(f"Error processing loitering detection database batch: {str(e)}")
    
    def save_loitering_person(self, person_id, bbox, frame, accumulated_time, camera_id):
        """Save loitering person screenshot and add to database"""
        try:
            # Create a filename for the incident image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            person_image_path = f"loitering_{person_id}_{timestamp}.jpg"
            
            # Create a copy of the frame for annotation
            annotated_frame = frame.copy()
            
            # Draw bounding box on the annotated frame
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add text label
            cv2.putText(annotated_frame, 
                    f"LOITERING: {accumulated_time:.1f}s",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save the annotated frame
            cv2.imwrite(person_image_path, annotated_frame)
            
            # Add to queue for database operations instead of starting a new thread
            self.db_queue.append((person_id, bbox, accumulated_time, camera_id, person_image_path, self.frame_count))
            
            return person_image_path
            
        except Exception as e:
            logger.error(f"Error saving loitering person: {str(e)}")
            return None
    
    def _add_to_database(self, person_id, bbox, accumulated_time, camera_id, image_path, frame_number):
        """Legacy method kept for compatibility - now adds to queue instead"""
        self.db_queue.append((person_id, bbox, accumulated_time, camera_id, image_path, frame_number))
        return True

    def detect(self, frame, camera_id="default", threshold_time=10, draw_visualization=True):
        """
        Perform loitering detection on a single frame
        Args:
            frame: Input video frame
            camera_id: Camera identifier
            threshold_time: Time threshold to consider as loitering
            draw_visualization: Whether to draw visualization on the frame
        Returns:
            Detection results with loitering incidents
        """
        try:
            if frame is None or self.object_model is None:
                return {"detected": False, "duration": 0, "regions": []}
            
            # Initialize time tracking for first frame
            current_time = time.time()
            if self.last_frame_time is None:
                self.last_frame_time = current_time
                
            # Calculate time between frames
            time_diff = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Increment frame counter
            self.frame_count += 1
            
            # Process only on specified interval frames for efficiency
            current_detections = []
            loitering_incidents = []
            
            # Only do detection on interval frames
            should_process = self.frame_count % self.process_interval == 0
            
            if should_process:
                # Convert frame to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Optimize inference parameters for 640p
                conf_threshold = 0.35  # Lower confidence threshold for faster processing with 640p
                iou_threshold = 0.5  # Higher IoU threshold to filter duplicates
                
                # Use half precision with RTX 4090 for better performance
                if CUDA_AVAILABLE and self.device == 0:
                    with torch.cuda.amp.autocast():
                        results = self.object_model(
                            frame_rgb, 
                            conf=conf_threshold,
                            iou=iou_threshold,
                            device=self.device,
                            verbose=False
                        )
                else:
                    results = self.object_model(
                        frame_rgb, 
                        conf=conf_threshold,
                        iou=iou_threshold,
                        device=self.device,
                        verbose=False
                    )
                
                # Process detected people
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Class 0 is 'person' in COCO dataset used by YOLO
                        class_name = self.object_model.names[int(cls_id)]
                        if class_name == 'person':
                            bbox = box.tolist()
                            
                            # Extract features for tracking
                            features = self.extract_person_features(frame, bbox)
                            
                            # Match with existing tracked persons
                            matched_id = self.match_person(features, bbox, self.frame_count)
                            
                            if matched_id is not None:
                                # Update existing person
                                person_id = matched_id
                                person_data = self.tracked_persons[person_id]
                                
                                # Update tracking info
                                person_data.update({
                                    'bbox': bbox,
                                    'features': features,
                                    'last_seen_frame': self.frame_count,
                                    'last_seen_time': current_time
                                })
                                
                                # Calculate time in frame
                                time_in_frame = current_time - person_data['first_seen_time']
                                
                                # Track accumulated time
                                if 'accumulated_time' not in person_data:
                                    person_data['accumulated_time'] = 0
                                
                                # Add time spent since last frame
                                person_data['accumulated_time'] += time_diff
                                
                                accumulated_time = person_data['accumulated_time']
                                
                                # Add to current detections
                                current_detections.append({
                                    'id': person_id,
                                    'bbox': bbox,
                                    'time_in_frame': time_in_frame,
                                    'accumulated_time': accumulated_time,
                                    'confidence': float(conf)
                                })
                                
                                # Check if person has been loitering
                                if accumulated_time > threshold_time:
                                    # Check if this loitering has already been detected
                                    if person_id not in self.loitering_incidents:
                                        # Save incident screenshot
                                        image_path = self.save_loitering_person(
                                            person_id, bbox, frame, accumulated_time, camera_id
                                        )
                                        
                                        # Add to incidents list
                                        loitering_incidents.append({
                                            "type": "loitering",
                                            "person_id": person_id,
                                            "confidence": float(conf),
                                            "bbox": bbox,
                                            "duration": accumulated_time,
                                            "image_path": image_path
                                        })
                            else:
                                # New person detected
                                person_id = f"person_{self.next_person_id}"
                                self.next_person_id += 1
                                
                                self.tracked_persons[person_id] = {
                                    'bbox': bbox,
                                    'features': features,
                                    'first_seen_frame': self.frame_count,
                                    'first_seen_time': current_time,
                                    'last_seen_frame': self.frame_count,
                                    'last_seen_time': current_time,
                                    'accumulated_time': 0
                                }
                                
                                # Add to current detections
                                current_detections.append({
                                    'id': person_id,
                                    'bbox': bbox,
                                    'time_in_frame': 0,
                                    'accumulated_time': 0,
                                    'confidence': float(conf)
                                })
            else:
                # For non-processing frames, just update time in tracked persons
                for person_id, person_data in self.tracked_persons.items():
                    if 'active' in person_data and person_data['active']:
                        if 'accumulated_time' in person_data:
                            person_data['accumulated_time'] += time_diff
            
            # Draw visualization on frame only if requested
            if draw_visualization:
                for person_id, person_data in self.tracked_persons.items():
                    if self.frame_count - person_data.get('last_seen_frame', 0) > 30:
                        continue
                        
                    bbox = person_data.get('bbox')
                    if not bbox:
                        continue
                        
                    accumulated_time = person_data.get('accumulated_time', 0)
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Determine color based on time (green->yellow->red as time increases)
                    if accumulated_time > threshold_time:
                        # Red for loitering
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {person_id} - LOITERING", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Time: {accumulated_time:.1f}s", (x1, y1 - 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        # Calculate color from green to yellow based on time
                        ratio = min(1.0, accumulated_time / threshold_time)
                        g = int(255 * (1 - ratio))
                        r = int(255 * ratio)
                        color = (0, g, r)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Time: {accumulated_time:.1f}s", (x1, y1 - 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Clean up old tracked persons less frequently
            if self.frame_count % 300 == 0:  # Much less frequent cleanup
                current_time = time.time()
                for person_id in list(self.tracked_persons.keys()):
                    last_seen_time = self.tracked_persons[person_id].get('last_seen_time', 0)
                    # Remove if not seen in the last 10 seconds 
                    if current_time - last_seen_time > 10:
                        del self.tracked_persons[person_id]
            
            return {
                "detected": len(loitering_incidents) > 0,
                "duration": max([person_data.get('accumulated_time', 0) for person_id, person_data in self.tracked_persons.items()]) if self.tracked_persons else 0,
                "regions": current_detections,
                "incidents": loitering_incidents,
                "tracked_persons": len(self.tracked_persons)
            }
            
        except Exception as e:
            logger.error(f"Error in loitering detection: {str(e)}")
            return {"detected": False, "duration": 0, "regions": [], "error": str(e)}

    def _load_model(self):
        """Load the object detection model"""
        try:
            # Import and load model (YOLO or similar)
            from ultralytics import YOLO
            # Use small model for better performance
            model_path = "yolov8n.pt"
            self.object_model = YOLO(model_path)
            
            # Use FP16 precision on RTX 4090
            if CUDA_AVAILABLE:
                logger.info(f"Using GPU acceleration for object model on {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Using CPU for object model")
        except Exception as e:
            logger.error(f"Failed to load object detection model: {str(e)}")
            self.object_model = None

video_processor = VideoProcessor()