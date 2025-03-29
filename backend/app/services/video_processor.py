import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import cv2
import numpy as np
import torch
import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from ..core.config import settings
from ..core.websocket import manager
from ..models.schemas import DetectionInfo, IncidentInfo
import json

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.face_model = None
        self.pose_model = None
        self.object_model = None
        self.face_extraction_model = None
        self.loitering_model = None
        self.theft_model = None
        self._initialize_models()
        
        # Initialize tracking dictionaries for theft detection
        self.hand_timers = {}
        self.previous_hand_positions = {}
        
        # COCO Pose Keypoints indices
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.NOSE = 0  # Used for front/back detection
        
        # Initialize DeepSort tracker if available
        self.tracker = None
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(max_age=30, nn_budget=100, override_track_class=None)
            logger.info("DeepSort tracker initialized successfully")
        except ImportError:
            logger.warning("DeepSort not installed. Person tracking will be limited.")

    def _initialize_models(self):
        """Initialize all required models"""
        try:
            from ultralytics import YOLO
            
            if settings.FACE_MODEL_PATH.exists():
                self.face_model = YOLO(str(settings.FACE_MODEL_PATH), task='detect')
                logger.info("Face detection model loaded successfully")
            
            if settings.POSE_MODEL_PATH.exists():
                self.pose_model = YOLO(str(settings.POSE_MODEL_PATH), task='detect')
                logger.info("Pose detection model loaded successfully")
            
            if settings.OBJECT_MODEL_PATH.exists():
                self.object_model = YOLO(str(settings.OBJECT_MODEL_PATH), task='detect')
                logger.info("Object detection model loaded successfully")
            
            if settings.FACE_EXTRACTION_MODEL_PATH.exists():
                self.face_extraction_model = YOLO(str(settings.FACE_EXTRACTION_MODEL_PATH))
                logger.info("Face extraction model loaded successfully")
            
            if settings.LOITERING_MODEL_PATH.exists():
                self.loitering_model = YOLO(str(settings.LOITERING_MODEL_PATH))
                logger.info("Loitering detection model loaded successfully")
            
            if settings.THEFT_MODEL_PATH.exists():
                self.theft_model = YOLO(str(settings.THEFT_MODEL_PATH))
                logger.info("Theft detection model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    async def process_frame(self, frame: np.ndarray, detection_type: str = "all") -> List[Dict]:
        """Process a single frame and return detections"""
        detections = []
        
        try:
            # Convert frame to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to CUDA tensor for faster processing
            if torch.cuda.is_available():
                frame_tensor = torch.from_numpy(frame_rgb).cuda()
            else:
                frame_tensor = torch.from_numpy(frame_rgb)
            
            if detection_type in ["all", "face"] and self.face_model:
                face_results = self.face_model(frame_rgb, conf=0.5, device=0)
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
                object_results = self.object_model(frame_rgb, conf=0.5, device=0)
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
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return []

    def draw_skeleton(self, frame, keypoints, color=(0, 255, 0), thickness=2):
        """
        Draw a skeleton on the frame based on keypoints.
        """
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
            (5, 6), (5, 7), (6, 8),          # Torso and arms
            (7, 9), (8, 10),                 # Arms
            (5, 11), (6, 12),                # Torso to legs
            (11, 12),                        # Hips
            (11, 13), (12, 14),              # Legs
            (13, 15), (14, 16)               # Ankles
        ]
        
        for keypoint in keypoints:
            if keypoint[0] > 0 and keypoint[1] > 0:  # Only draw visible keypoints
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)
                
        for connection in connections:
            idx1, idx2 = connection
            if (idx1 < len(keypoints) and idx2 < len(keypoints) and 
                keypoints[idx1][0] > 0 and keypoints[idx1][1] > 0 and 
                keypoints[idx2][0] > 0 and keypoints[idx2][1] > 0):
                pt1 = tuple(map(int, keypoints[idx1][:2]))
                pt2 = tuple(map(int, keypoints[idx2][:2]))
                cv2.line(frame, pt1, pt2, color, thickness)
    
    def approximate_eye_gaze(self, nose, chest_box):
        """
        Approximate whether the person is looking at their chest region using the nose keypoint.
        """
        if nose is None or nose[0] <= 0 or nose[1] <= 0:
            return False
        nose_x, nose_y = int(nose[0]), int(nose[1])
        return (chest_box[0] < nose_x < chest_box[2]) and (chest_box[1] < nose_y < chest_box[3])

    def is_intersecting(self, box1, box2):
        """Check if two bounding boxes intersect"""
        return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])
    
    async def detect_theft(self, frame: np.ndarray, frame_count: int, video_id: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect potential theft activities in the frame
        Returns: Modified frame with annotations and list of theft detections
        """
        theft_detections = []
        frame_height, frame_width = frame.shape[:2]
        
        try:
            if self.theft_model is None:
                logger.warning("Theft detection model not loaded")
                return frame, []
            
            # Run pose estimation
            pose_results = self.theft_model(frame)
            
            detections_for_tracker = []
            
            for result in pose_results:
                keypoints = result.keypoints.xy.cpu().numpy() if hasattr(result, 'keypoints') and result.keypoints is not None else []
                boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, 'boxes') and result.boxes is not None else []
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    width, height = x2 - x1, y2 - y1
                    
                    # Add detection to tracker if available
                    person_box = [x1, y1, x2 - x1, y2 - y1]  # Format: [x, y, width, height]
                    detections_for_tracker.append((person_box, 1.0, "person"))
                    
                    if len(keypoints) <= i:
                        continue
                    
                    person_keypoints = keypoints[i]
                    
                    # Define chest and waist bounding boxes for theft detection zones
                    chest_box = [x1 + int(0.1 * width), y1, x2 - int(0.1 * width), y1 + int(0.4 * height)]
                    left_waist_box = [x1, y1 + int(0.5 * height), x1 + int(0.5 * width), y2]
                    right_waist_box = [x1 + int(0.5 * width), y1 + int(0.5 * height), x2, y2]
                    
                    # Draw detection zones
                    cv2.rectangle(frame, tuple(chest_box[:2]), tuple(chest_box[2:]), (0, 255, 255), 2)
                    cv2.rectangle(frame, tuple(left_waist_box[:2]), tuple(left_waist_box[2:]), (255, 0, 0), 2)
                    cv2.rectangle(frame, tuple(right_waist_box[:2]), tuple(right_waist_box[2:]), (0, 0, 255), 2)
                    
                    # Draw skeleton
                    self.draw_skeleton(frame, person_keypoints)
                    
                    # Get wrist and nose positions
                    left_wrist = person_keypoints[self.LEFT_WRIST] if len(person_keypoints) > self.LEFT_WRIST and person_keypoints[self.LEFT_WRIST][0] > 0 else None
                    right_wrist = person_keypoints[self.RIGHT_WRIST] if len(person_keypoints) > self.RIGHT_WRIST and person_keypoints[self.RIGHT_WRIST][0] > 0 else None
                    nose = person_keypoints[self.NOSE] if len(person_keypoints) > self.NOSE and person_keypoints[self.NOSE][0] > 0 else None
                    
                    # Approximate eye gaze direction
                    is_looking_at_chest = self.approximate_eye_gaze(nose, chest_box)
                    
                    # Track hands in suspicious regions (chest/waist)
                    for wrist, label in [(left_wrist, f"left_{i}"), (right_wrist, f"right_{i}")]:
                        if wrist is None:
                            continue
                        
                        wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                        wrist_box = [wrist_x - 10, wrist_y - 10, wrist_x + 10, wrist_y + 10]
                        cv2.rectangle(frame, (wrist_box[0], wrist_box[1]), (wrist_box[2], wrist_box[3]), (0, 255, 0), 2)
                        
                        in_chest = self.is_intersecting(wrist_box, chest_box)
                        in_left_waist = self.is_intersecting(wrist_box, left_waist_box)
                        in_right_waist = self.is_intersecting(wrist_box, right_waist_box)
                        
                        # Suspicious behavior detection
                        hand_stay_time = settings.HAND_STAY_TIME_WAIST if in_left_waist or in_right_waist else settings.HAND_STAY_TIME_CHEST
                        
                        if in_chest or in_left_waist or in_right_waist:
                            current_time = time.time()
                            if label not in self.hand_timers:
                                self.hand_timers[label] = current_time
                            elif current_time - self.hand_timers[label] > hand_stay_time:
                                if not is_looking_at_chest:  # Increase suspicion if not looking at chest
                                    cv2.putText(frame, "⚠️ Shoplifter!", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    
                                    # Crop the suspect with padding
                                    x1_crop = max(0, x1 - settings.CROP_PADDING)
                                    y1_crop = max(0, y1 - settings.CROP_PADDING)
                                    x2_crop = min(frame_width, x2 + settings.CROP_PADDING)
                                    y2_crop = min(frame_height, y2 + settings.CROP_PADDING)
                                    suspect_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                                    
                                    # Save screenshot
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    screenshot_path = settings.SCREENSHOTS_DIR / f"theft_{video_id}_{timestamp}_{frame_count}.jpg"
                                    os.makedirs(settings.SCREENSHOTS_DIR, exist_ok=True)
                                    cv2.imwrite(str(screenshot_path), suspect_crop)
                                    
                                    # Add to detections
                                    theft_detections.append({
                                        "type": "theft",
                                        "confidence": 0.85,  # Confidence score for theft detection
                                        "bbox": [x1, y1, x2, y2],
                                        "screenshot": str(screenshot_path),
                                        "frame_number": frame_count,
                                        "timestamp": timestamp
                                    })
                            
                            self.previous_hand_positions[label] = (in_chest, in_left_waist, in_right_waist)
                        else:
                            self.hand_timers.pop(label, None)
            
            # Update tracker if available
            if self.tracker is not None and detections_for_tracker:
                tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    track_id = track.track_id
                    bbox = track.to_tlbr()  # Get bounding box in [x1, y1, x2, y2] format
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Display person ID
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame, theft_detections
            
        except Exception as e:
            logger.error(f"Error in theft detection: {str(e)}")
            return frame, []

    async def process_video(self, video_path: str, video_id: int, detection_type: str) -> None:
        """Process video and store detections"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            output_path = settings.PROCESSED_DIR / f"processed_{video_id}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_number = 0
            detections = []
            theft_incidents = []
            
            # Create directories if they don't exist
            os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
            os.makedirs(settings.SCREENSHOTS_DIR, exist_ok=True)
            os.makedirs(settings.THUMBNAILS_DIR, exist_ok=True)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame based on settings
                if frame_number % settings.SKIP_FRAMES == 0:
                    # Regular object/face detection
                    if detection_type in ["all", "face", "object"]:
                        frame_detections = await self.process_frame(frame, detection_type)
                        detections.extend(frame_detections)
                        
                        # Draw detections on frame
                        for det in frame_detections:
                            x1, y1, x2, y2 = map(int, det["bbox"])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{det['class_name']} {det['confidence']:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Theft detection
                    if detection_type in ["all", "theft"] and self.theft_model:
                        frame, theft_detections = await self.detect_theft(frame, frame_number, video_id)
                        
                        if theft_detections:
                            theft_incidents.extend(theft_detections)
                            
                            # Send real-time alert via websocket
                            for theft in theft_detections:
                                await manager.broadcast(json.dumps({
                                    "type": "theft_alert",
                                    "video_id": video_id,
                                    "frame": frame_number,
                                    "bbox": theft["bbox"],
                                    "screenshot": theft["screenshot"],
                                    "timestamp": theft["timestamp"]
                                }))
                
                # Write processed frame
                out.write(frame)
                frame_number += 1
                
                # Update progress
                if frame_number % 10 == 0:  # Update every 10 frames to reduce overhead
                    progress = (frame_number / frame_count) * 100
                    await manager.broadcast(json.dumps({
                        "type": "processing_progress",
                        "video_id": video_id,
                        "progress": progress
                    }))
            
            cap.release()
            out.release()
            
            # Generate thumbnail (using the last frame)
            thumbnail_path = settings.THUMBNAILS_DIR / f"thumb_{video_id}.jpg"
            cv2.imwrite(str(thumbnail_path), frame)
            
            # Combine all detections
            all_detections = detections + theft_incidents
            
            return {
                "output_path": str(output_path),
                "thumbnail_path": str(thumbnail_path),
                "detections": all_detections,
                "theft_incidents": theft_incidents
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

video_processor = VideoProcessor()