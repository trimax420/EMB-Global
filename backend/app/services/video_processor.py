import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import cv2
import numpy as np
import torch
import asyncio
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from ..core.config import settings
from ..core.websocket import manager

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.face_model = None
        self.pose_model = None
        self.object_model = None
        self._initialize_models()
        
        # COCO Pose Keypoints indices for reference
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.NOSE = 0
        
        # Initialize tracker if available
        self.tracker = None
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(max_age=30, nn_budget=100)
            logger.info("DeepSort tracker initialized")
        except ImportError:
            logger.warning("DeepSort not installed. Object tracking will be limited.")

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
            raise

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
        """Process video and return detections"""
        try:
            # Ensure directories exist
            os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
            os.makedirs(settings.THUMBNAILS_DIR, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
                
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video writer
            output_path = settings.PROCESSED_DIR / f"processed_{video_id}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Process frames
            frame_number = 0
            detections = []
            thumbnail_saved = False
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame to improve performance
                if frame_number % settings.SKIP_FRAMES == 0:
                    # Get detections for the current frame
                    frame_detections = await self.process_frame(frame, detection_type)
                    
                    # Add frame number to detections
                    for det in frame_detections:
                        det["frame_number"] = frame_number
                        detections.append(det)
                    
                    # Draw detections on frame
                    for det in frame_detections:
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Save thumbnail if not yet saved
                    if not thumbnail_saved:
                        thumbnail_path = settings.THUMBNAILS_DIR / f"thumb_{video_id}.jpg"
                        cv2.imwrite(str(thumbnail_path), frame)
                        thumbnail_saved = True
                
                # Write processed frame
                out.write(frame)
                frame_number += 1
                
                # Report progress periodically
                if frame_number % 30 == 0:  # Update every 30 frames
                    progress = (frame_number / total_frames) * 100
                    await manager.broadcast(json.dumps({
                        "type": "processing_progress",
                        "video_id": video_id,
                        "progress": progress
                    }))
            
            # Release resources
            cap.release()
            out.release()
            
            # If we didn't save a thumbnail yet, use the last frame
            if not thumbnail_saved and frame is not None:
                thumbnail_path = settings.THUMBNAILS_DIR / f"thumb_{video_id}.jpg"
                cv2.imwrite(str(thumbnail_path), frame)
            
            return {
                "output_path": str(output_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_saved else None,
                "detections": detections,
                "total_frames": frame_number,
                "fps": fps,
                "resolution": f"{width}x{height}"
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    async def process_face_extraction(self, video_path, save_path, confidence_threshold=0.5):
        """Extract faces from a video and save them as individual images"""
        try:
            logger.info(f"Starting face extraction on {video_path}")
            
            if self.face_model is None:
                raise ValueError("Face detection model not loaded")
            
            # Ensure save directory exists
            os.makedirs(save_path, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            face_count = 0
            
            # Process frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every N frames
                if frame_count % settings.SKIP_FRAMES == 0:
                    # Detect faces in the frame
                    device = 0 if torch.cuda.is_available() else 'cpu'
                    results = self.face_model(frame, conf=confidence_threshold, device=device)
                    
                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            
                            for box, conf in zip(boxes, confidences):
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Add padding around face
                                padding = int((x2 - x1) * 0.2)
                                x1 = max(0, x1 - padding)
                                y1 = max(0, y1 - padding)
                                x2 = min(frame.shape[1], x2 + padding)
                                y2 = min(frame.shape[0], y2 + padding)
                                
                                # Extract face ROI
                                face_roi = frame[y1:y2, x1:x2]
                                
                                if face_roi.size == 0:
                                    continue
                                
                                # Save face image
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                face_path = os.path.join(save_path, f"face_{timestamp}_{face_count}.jpg")
                                cv2.imwrite(face_path, face_roi)
                                face_count += 1
                
                frame_count += 1
                
                # Report progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Face extraction progress: {progress:.1f}%, extracted {face_count} faces")
            
            # Release resources
            cap.release()
            
            logger.info(f"Face extraction completed: extracted {face_count} face images")
            
            return {
                "save_path": save_path,
                "total_faces": face_count,
                "total_frames_processed": frame_count
            }
            
        except Exception as e:
            logger.error(f"Error in face extraction: {str(e)}")
            raise

    async def process_theft_detection(self, video_path, output_path, screenshot_dir, hand_stay_time=2):
        """Process video for theft detection"""
        try:
            logger.info(f"Starting theft detection on {video_path}")
            
            if self.pose_model is None:
                raise ValueError("Pose detection model not loaded")
            
            # Ensure output directories exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.makedirs(screenshot_dir, exist_ok=True)
            
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
            
            # Initialize tracking variables
            hand_timers = {}
            
            frame_count = 0
            theft_detections = []
            device = 0 if torch.cuda.is_available() else 'cpu'
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every N frames
                if frame_count % settings.SKIP_FRAMES == 0:
                    # Run pose detection
                    results = self.pose_model(frame, conf=0.5, device=device)
                    
                    for result in results:
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            keypoints = result.keypoints.xy.cpu().numpy()
                            boxes = result.boxes.xyxy.cpu().numpy()
                            
                            for i, box in enumerate(boxes):
                                x1, y1, x2, y2 = map(int, box)
                                person_id = f"person_{i}"
                                
                                # Draw person bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Define chest region
                                chest_y_top = y1 + int((y2 - y1) * 0.2)
                                chest_y_bottom = y1 + int((y2 - y1) * 0.4)
                                chest_box = [x1, chest_y_top, x2, chest_y_bottom]
                                
                                # Draw chest region
                                cv2.rectangle(frame, (chest_box[0], chest_box[1]), (chest_box[2], chest_box[3]), (0, 255, 255), 2)
                                
                                # Check if person has keypoints
                                if i < len(keypoints):
                                    person_kpts = keypoints[i]
                                    
                                    # Get wrist keypoints
                                    left_wrist = person_kpts[self.LEFT_WRIST] if len(person_kpts) > self.LEFT_WRIST else None
                                    right_wrist = person_kpts[self.RIGHT_WRIST] if len(person_kpts) > self.RIGHT_WRIST else None
                                    
                                    # Check for hand near chest
                                    for wrist, hand_id in [(left_wrist, f"left_{i}"), (right_wrist, f"right_{i}")]:
                                        if wrist is not None and wrist[0] > 0 and wrist[1] > 0:
                                            wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                                            
                                            # Check if wrist is in chest region
                                            in_chest = (chest_box[0] <= wrist_x <= chest_box[2] and 
                                                       chest_box[1] <= wrist_y <= chest_box[3])
                                            
                                            if in_chest:
                                                # Start timer for hand in chest
                                                current_time = time.time()
                                                if hand_id not in hand_timers:
                                                    hand_timers[hand_id] = current_time
                                                elif current_time - hand_timers[hand_id] > hand_stay_time:
                                                    # Suspiciously long hand stay in chest region
                                                    cv2.putText(frame, "Suspicious Activity", (x1, y1 - 10),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                                    
                                                    # Save a screenshot
                                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                    ss_path = os.path.join(screenshot_dir, f"theft_{timestamp}_{frame_count}.jpg")
                                                    cv2.imwrite(ss_path, frame)
                                                    
                                                    # Record detection
                                                    theft_detections.append({
                                                        "type": "theft",
                                                        "confidence": 0.8,
                                                        "bbox": [x1, y1, x2, y2],
                                                        "frame_number": frame_count,
                                                        "screenshot": ss_path
                                                    })
                                            else:
                                                # Hand in chest but not long enough yet
                                                cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 255, 255), -1)
                                        else:
                                            # Hand not in chest, reset timer
                                            if hand_id in hand_timers:
                                                del hand_timers[hand_id]
                
                # Write processed frame
                out.write(frame)
                
                frame_count += 1
                
                # Report progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Theft detection progress: {progress:.1f}%")
            
            # Release resources
            cap.release()
            out.release()
            
            return {
                "output_path": output_path,
                "screenshot_dir": screenshot_dir,
                "detections": theft_detections,
                "total_frames": frame_count
            }
        
        except Exception as e:
            logger.error(f"Error in theft detection: {str(e)}")
            raise

    async def process_loitering_detection(self, video_path, output_path, threshold_time=10):
        """Process video for loitering detection"""
        try:
            logger.info(f"Starting loitering detection on {video_path}")
            
            if self.object_model is None:
                raise ValueError("Object detection model not loaded")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
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
            
            # Initialize people tracking
            person_trackers = {}  # {id: {'first_seen': time, 'position': [x, y]}}
            device = 0 if torch.cuda.is_available() else 'cpu'
            
            frame_count = 0
            loitering_detections = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every N frames
                if frame_count % settings.SKIP_FRAMES == 0:
                    # Current time in video
                    current_time = frame_count / fps
                    
                    # Detect people
                    results = self.object_model(frame, conf=0.5, device=device)
                    
                    # Extract person detections
                    detections_for_tracker = []
                    
                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy()
                            
                            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                                class_name = self.object_model.names[int(cls_id)]
                                
                                # Only track people
                                if class_name == "person" and conf > 0.5:
                                    x1, y1, x2, y2 = map(int, box)
                                    width = x2 - x1
                                    height = y2 - y1
                                    
                                    # Add to tracker format: [x, y, width, height]
                                    detections_for_tracker.append(([x1, y1, width, height], conf, class_name))
                    
                    # Update tracker
                    if self.tracker is not None and detections_for_tracker:
                        tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)
                        
                        for track in tracks:
                            if not track.is_confirmed():
                                continue
                            
                            track_id = track.track_id
                            track_bbox = track.to_tlbr()  # [x1, y1, x2, y2]
                            x1, y1, x2, y2 = map(int, track_bbox)
                            
                            # Get center position
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            # Track person
                            if track_id not in person_trackers:
                                person_trackers[track_id] = {
                                    'first_seen': current_time,
                                    'position': [center_x, center_y]
                                }
                            else:
                                # Calculate time present
                                time_present = current_time - person_trackers[track_id]['first_seen']
                                
                                # Update position
                                person_trackers[track_id]['position'] = [center_x, center_y]
                                
                                # Check for loitering
                                if time_present > threshold_time:
                                    # Person has been present too long
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(frame, f"Loitering: {time_present:.1f}s", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    
                                    # Add loitering detection (once per 2 seconds)
                                    if int(time_present) % 2 == 0 and int(time_present) != int(time_present - 1):
                                        loitering_detection = {
                                            "type": "loitering",
                                            "track_id": track_id,
                                            "confidence": 0.9,
                                            "bbox": [x1, y1, x2, y2],
                                            "time_present": time_present,
                                            "frame_number": frame_count
                                        }
                                        loitering_detections.append(loitering_detection)
                                else:
                                    # Normal tracking
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Person {track_id}: {time_present:.1f}s", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Write processed frame
                out.write(frame)
                
                frame_count += 1
                
                # Report progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Loitering detection progress: {progress:.1f}%")
            
            # Release resources
            cap.release()
            out.release()
            
            return {
                "output_path": output_path,
                "detections": loitering_detections,
                "total_frames": frame_count
            }
            
        except Exception as e:
            logger.error(f"Error in loitering detection: {str(e)}")
            raise

video_processor = VideoProcessor()