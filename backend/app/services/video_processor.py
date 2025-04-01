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

    async def process_loitering_detection(self, video_path, output_path, camera_id=None, threshold_time=10):
        """
        Process video for loitering detection with improved person re-identification.
        Saves person ID, image, loitering time, and camera ID to the database.
        Only one record per person is maintained (updated with the most recent data).
        
        Args:
            video_path: Path to input video file
            output_path: Path to save processed video
            camera_id: ID of the camera feed (if available)                        
            threshold_time: Time threshold (in seconds) to consider loitering
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Starting loitering detection on {video_path}")
            
            if self.object_model is None:
                raise ValueError("Object detection model not loaded")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create directory for loitering screenshots
            screenshots_dir = os.path.join(settings.SCREENSHOTS_DIR, f"loitering_{int(time.time())}")
            os.makedirs(screenshots_dir, exist_ok=True)
            
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
            
            # Initialize people tracking with persistent features for re-identification
            person_features = {}  # Dictionary to store person features for re-identification
            person_trackers = {}  # {person_id: {'first_seen': time, 'position': [x, y], 'accumulated_time': time, 'loitering_detected': bool}}
            loitering_db_entries = {}  # {person_id: {'db_id': id, 'last_updated': timestamp}}
            person_last_frame = {}  # Store the last valid frame for each person
            
            # Define helper functions
            def extract_person_features(frame, bbox):
                """Extract color histogram features from person region"""
                x1, y1, x2, y2 = map(int, bbox)
                # Ensure bbox is within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    return None  # Invalid bbox
                    
                person_roi = frame[y1:y2, x1:x2]
                # Convert to HSV colorspace for better feature representation
                hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
                # Calculate histogram
                hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
                # Normalize the histogram
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                return hist.flatten()
            
            def match_person(features, existing_features, threshold=0.7):
                """Match person features against existing person features"""
                best_match_id = None
                best_match_score = 0
                
                for person_id, person_feat in existing_features.items():
                    if person_trackers[person_id]['active'] == False:  # Only consider inactive tracks
                        # Compare histograms using correlation
                        score = cv2.compareHist(
                            features.reshape(-1, 1),
                            person_feat.reshape(-1, 1),
                            cv2.HISTCMP_CORREL
                        )
                        
                        if score > threshold and score > best_match_score:
                            best_match_score = score
                            best_match_id = person_id
                
                return best_match_id
            
            def get_valid_bbox(position, frame_width, frame_height):
                """Create a valid bounding box within frame dimensions"""
                # Create a bounding box centered at the position with reasonable default size
                box_size = 100  # Default size
                x1 = max(0, int(position[0] - box_size/2))
                y1 = max(0, int(position[1] - box_size/2))
                x2 = min(frame_width, int(position[0] + box_size/2))
                y2 = min(frame_height, int(position[1] + box_size/2))
                
                # Ensure box has minimum size
                if x2 - x1 < 10:
                    x2 = min(frame_width, x1 + 10)
                if y2 - y1 < 10:
                    y2 = min(frame_height, y1 + 10)
                    
                return np.array([x1, y1, x2, y2])
            
            # Helper function to save/update loitering person in database
            async def save_or_update_loitering_person(person_id, bbox, frame, accumulated_time, camera_id):
                """Save or update loitering person in the database"""
                if frame is None:
                    logger.warning(f"Frame is None for person {person_id}, using stored frame if available")
                    # Try to use a stored frame for this person
                    if person_id in person_last_frame:
                        frame = person_last_frame[person_id]
                    else:
                        logger.error(f"No stored frame available for person {person_id}, cannot save/update")
                        return None
                
                x1, y1, x2, y2 = map(int, bbox)
                # Ensure bbox is within frame boundaries
                frame_height, frame_width = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bounding box for person {person_id}: {bbox}")
                    return None
                    
                current_time = datetime.now()
                
                # Create a file path for the person's image
                person_image_filename = f"loitering_{person_id}_{int(time.time())}.jpg"
                person_image_path = os.path.join(screenshots_dir, person_image_filename)
                
                try:
                    # Crop and save the person image
                    person_roi = frame[y1:y2, x1:x2]
                    cv2.imwrite(person_image_path, person_roi)
                    
                    # Format the incident data
                    incident_data = {
                        "type": "loitering",
                        "timestamp": current_time,
                        "location": f"Camera {camera_id}" if camera_id else "Unknown",
                        "description": f"Person loitering for {accumulated_time:.1f} seconds",
                        "image_path": person_image_path,
                        "video_url": video_path,
                        "severity": "medium",
                        "confidence": 0.9,
                        "duration": accumulated_time,
                        "is_resolved": False,
                    }
                    
                    # For incident creation or update, also include detection metadata
                    detection_data = {
                        "video_id": None,  # Live feed doesn't have a video ID
                        "camera_id": camera_id,
                        "timestamp": current_time,
                        "frame_number": frame_count,
                        "detection_type": "loitering",
                        "confidence": 0.9,
                        "bbox": bbox.tolist(),
                        "class_name": "person",
                        "image_path": person_image_path,
                        "detection_metadata": {
                            "person_id": person_id,
                            "accumulated_time": accumulated_time,
                            "first_detected": person_trackers[person_id]['first_seen'],
                        }
                    }

                    try:
                        from database import add_detection, add_incident, update_incident
                        
                        if person_id in loitering_db_entries:
                            # Update existing entry with latest information
                            incident_id = loitering_db_entries[person_id]['db_id']
                            
                            # Add new detection
                            detection_id = await add_detection(detection_data)
                            
                            # Update incident with latest information
                            update_data = {
                                "duration": accumulated_time,
                                "description": f"Person loitering for {accumulated_time:.1f} seconds",
                                "image_path": person_image_path,  # Update with latest image
                                "detection_ids": [detection_id],  # Add new detection ID
                            }
                            await update_incident(incident_id, update_data)
                            
                            # Update timestamp of last database update
                            loitering_db_entries[person_id]['last_updated'] = current_time
                            logger.info(f"Updated loitering record for {person_id} in database")
                        else:
                            # Create new detection entry
                            detection_id = await add_detection(detection_data)
                            
                            # Include detection ID in incident
                            incident_data["detection_ids"] = [detection_id]
                            
                            # Create new incident entry
                            incident_id = await add_incident(incident_data)
                            
                            # Record database entry for future updates
                            loitering_db_entries[person_id] = {
                                'db_id': incident_id,
                                'last_updated': current_time
                            }
                            logger.info(f"Created new loitering record for {person_id} in database")
                            
                        # Return the database record ID
                        return loitering_db_entries[person_id]['db_id']
                        
                    except Exception as e:
                        logger.error(f"Error saving loitering person to database: {str(e)}")
                        return None
                        
                except Exception as e:
                    logger.error(f"Error processing person roi: {str(e)}")
                    return None
            
            frame_count = 0
            loitering_detections = []
            device = 0 if torch.cuda.is_available() else 'cpu'
            next_person_id = 1  # Initial ID for new persons
            last_processed_frame = None  # Keep track of the last successfully processed frame
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Store the last successfully read frame
                last_processed_frame = frame.copy()
                
                # Process every N frames
                if frame_count % settings.SKIP_FRAMES == 0:
                    # Current time in video
                    current_time = frame_count / fps
                    
                    # Detect people
                    results = self.object_model(frame, conf=0.5, device=device)
                    
                    # Extract person detections
                    detections = []
                    
                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy()
                            
                            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                                class_name = self.object_model.names[int(cls_id)]
                                
                                # Only track people
                                if class_name == "person" and conf > 0.5:
                                    detections.append((box, conf))
                    
                    # Update active flags for all current persons
                    for person_id in person_trackers:
                        person_trackers[person_id]['active'] = False
                    
                    # Process each detected person
                    for bbox, conf in detections:
                        # Extract features for this person
                        features = extract_person_features(frame, bbox)
                        if features is None:
                            continue
                            
                        # Try to match with an existing person who is no longer tracked
                        matched_id = match_person(features, person_features)
                        
                        if matched_id is not None:
                            # Found a match - resume tracking
                            person_id = matched_id
                            # Update features with a moving average for better adaptive matching
                            person_features[person_id] = 0.7 * person_features[person_id] + 0.3 * features
                            # Mark as active again and preserve accumulated time
                            person_trackers[person_id]['active'] = True
                            logger.info(f"Re-identified person {person_id}")
                        else:
                            # New person detected
                            person_id = f"person_{next_person_id}"
                            next_person_id += 1
                            # Store features for future matching
                            person_features[person_id] = features
                            # Initialize tracking data
                            person_trackers[person_id] = {
                                'first_seen': current_time,
                                'last_seen': current_time,
                                'position': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                                'accumulated_time': 0,
                                'active': True,
                                'loitering_detected': False,
                                'loitering_frame_saved': False,
                                'db_record_created': False,
                                'last_db_update_time': 0  # Time threshold for database updates
                            }
                            logger.info(f"New person detected: {person_id}")
                        
                        # Store the current frame for this person
                        x1, y1, x2, y2 = map(int, bbox)
                        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x2 < frame.shape[1] and y2 < frame.shape[0]:
                            # Store a copy of the frame with this person
                            person_frame = frame.copy()
                            person_last_frame[person_id] = person_frame
                        
                        # Update person position and time
                        person_trackers[person_id]['position'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                        
                        # Update last seen time for active tracking
                        person_trackers[person_id]['last_seen'] = current_time
                        
                        # Calculate time present (accumulated time)
                        time_inactive = 0
                        if 'last_update' in person_trackers[person_id]:
                            time_inactive = current_time - person_trackers[person_id]['last_update']
                        
                        # Add time to accumulated total (but only if not gone for too long)
                        if time_inactive < 5:  # If person was gone for less than 5 seconds, consider it the same session
                            person_trackers[person_id]['accumulated_time'] += time_inactive
                        
                        person_trackers[person_id]['last_update'] = current_time
                        
                        # Check for loitering based on accumulated time
                        accumulated_time = person_trackers[person_id]['accumulated_time']
                        
                        # Draw bounding box and info
                        if accumulated_time > threshold_time:
                            # Person has been present too long (loitering)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"Loitering: {accumulated_time:.1f}s", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Mark as loitering
                            person_trackers[person_id]['loitering_detected'] = True
                            
                            # Save to database if not already done or if it's been a while since last update
                            current_real_time = time.time()
                            db_update_needed = False
                            
                            if not person_trackers[person_id]['db_record_created']:
                                # First time detecting loitering for this person
                                db_update_needed = True
                                person_trackers[person_id]['db_record_created'] = True
                            elif current_real_time - person_trackers[person_id].get('last_db_update_time', 0) > 30:
                                # Update database every 30 seconds if person continues loitering
                                db_update_needed = True
                            
                            if db_update_needed:
                                # Save/update in database
                                await save_or_update_loitering_person(
                                    person_id, 
                                    bbox, 
                                    frame, 
                                    accumulated_time, 
                                    camera_id
                                )
                                person_trackers[person_id]['last_db_update_time'] = current_real_time
                            
                            # Capture one frame per loitering person if not already saved (for local records)
                            if not person_trackers[person_id]['loitering_frame_saved']:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                screenshot_path = os.path.join(screenshots_dir, f"loitering_{person_id}_{timestamp}.jpg")
                                cv2.imwrite(screenshot_path, frame)
                                
                                # Record loitering detection once per person (for function return value)
                                loitering_detection = {
                                    "type": "loitering",
                                    "person_id": person_id,
                                    "confidence": 0.9,
                                    "bbox": bbox.tolist(),
                                    "time_present": accumulated_time,
                                    "frame_number": frame_count,
                                    "screenshot_path": screenshot_path,
                                    "camera_id": camera_id
                                }
                                loitering_detections.append(loitering_detection)
                                
                                # Mark that we've saved a frame for this person
                                person_trackers[person_id]['loitering_frame_saved'] = True
                                logger.info(f"Loitering detected for {person_id}, time: {accumulated_time:.1f}s")
                        else:
                            # Normal tracking
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{person_id}: {accumulated_time:.1f}s", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                    # Clean up old inactive tracks after a certain time to avoid memory issues
                    current_ids = list(person_trackers.keys())
                    for person_id in current_ids:
                        if (not person_trackers[person_id]['active'] and 
                            current_time - person_trackers[person_id]['last_seen'] > 60):  # Remove after 60 seconds inactive
                            # Keep the feature for re-identification but remove from active tracking
                            # This is a balance - keeping all features forever would be memory intensive
                            # So we keep them for recent individuals who might return
                            if person_trackers[person_id].get('loitering_detected', False):
                                # Keep features longer for people who were loitering
                                pass  # Don't remove them
                            else:
                                # For non-loitering people, we can remove them from memory after a while
                                if current_time - person_trackers[person_id]['last_seen'] > 300:  # 5 minutes
                                    if person_id in person_features:
                                        del person_features[person_id]
                                    if person_id in person_last_frame:
                                        del person_last_frame[person_id]
                                    del person_trackers[person_id]
                
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
            
            # Add final database updates for all loitering people
            try:
                for person_id, tracker in person_trackers.items():
                    if tracker.get('loitering_detected', False) and tracker.get('active', False):
                        # Get the last known position for this person
                        if person_id in person_last_frame:
                            frame = person_last_frame[person_id]
                            frame_height, frame_width = frame.shape[:2]
                            position = tracker['position']
                            
                            # Generate a valid bounding box
                            last_bbox = get_valid_bbox(position, frame_width, frame_height)
                            
                            try:
                                await save_or_update_loitering_person(
                                    person_id,
                                    last_bbox,
                                    frame,
                                    tracker['accumulated_time'],
                                    camera_id
                                )
                                logger.info(f"Final update for loitering person {person_id}")
                            except Exception as e:
                                logger.error(f"Error in final update for person {person_id}: {str(e)}")
            except Exception as e:
                logger.error(f"Error in final database updates: {str(e)}")
            
            return {
                "output_path": output_path,
                "screenshots_dir": screenshots_dir,
                "detections": loitering_detections,
                "total_frames": frame_count,
                "unique_persons": next_person_id - 1,
                "loitering_persons": [
                    {"person_id": p_id, "time": tracker['accumulated_time']}
                    for p_id, tracker in person_trackers.items()
                    if tracker.get('loitering_detected', False)
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in loitering detection: {str(e)}")
            raise

    async def process_theft_detection_fixed(self, video_path, output_path, screenshots_dir, hand_stay_time=2.0, camera_id=None):
        """
        Process video for theft detection using the original logic but with fixed tracking.
        Detects suspicious hand movements in chest and waist areas and monitors eye gaze.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save processed video
            screenshots_dir: Directory to save suspicious activity screenshots
            hand_stay_time: Default threshold time (in seconds) for suspicious hand positions
            camera_id: ID of the camera feed (if available)
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Starting fixed theft detection on {video_path}")
            
            if self.pose_model is None:
                raise ValueError("Pose detection model not loaded")
            
            # Ensure output directories exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.makedirs(screenshots_dir, exist_ok=True)
            
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
            
            # Define theft detection settings
            skip_frames = settings.SKIP_FRAMES
            hand_stay_time_chest = settings.HAND_STAY_TIME_CHEST
            hand_stay_time_waist = settings.HAND_STAY_TIME_WAIST
            crop_padding = settings.CROP_PADDING
            
            # COCO Pose Keypoints
            LEFT_WRIST = self.LEFT_WRIST
            RIGHT_WRIST = self.RIGHT_WRIST
            LEFT_SHOULDER = self.LEFT_SHOULDER
            RIGHT_SHOULDER = self.RIGHT_SHOULDER
            NOSE = self.NOSE
            
            # Tracking variables
            person_tracking = {}
            hand_timers = {}
            previous_hand_positions = {}
            next_person_id = 1
            detected_theft_persons = set()
            
            # Helper functions
            def is_intersecting(box1, box2):
                """Check if two boxes intersect"""
                return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])
                
            def approximate_eye_gaze(nose, chest_box):
                """Approximate whether the person is looking at their chest region using the nose keypoint."""
                if nose is None or nose[0] <= 0:
                    return False
                nose_x, nose_y = int(nose[0]), int(nose[1])
                return (chest_box[0] < nose_x < chest_box[2]) and (chest_box[1] < nose_y < chest_box[3])
                
            def draw_skeleton(frame, keypoints, color=(0, 255, 0), thickness=2):
                """Draw a skeleton on the frame based on keypoints."""
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
                    if idx1 < len(keypoints) and idx2 < len(keypoints):
                        if keypoints[idx1][0] > 0 and keypoints[idx1][1] > 0 and keypoints[idx2][0] > 0 and keypoints[idx2][1] > 0:
                            pt1 = tuple(map(int, keypoints[idx1][:2]))
                            pt2 = tuple(map(int, keypoints[idx2][:2]))
                            cv2.line(frame, pt1, pt2, color, thickness)
                            
            # Helper function to save theft incident to database
            async def save_theft_incident(person_id, frame, bbox, keypoints, zone_type, looking_at_chest=False):
                incident_id = None
                timestamp = datetime.now()
                
                try:
                    # Create a filename for the incident image
                    incident_filename = f"theft_{person_id}_{int(time.time())}.jpg"
                    incident_image_path = os.path.join(screenshots_dir, incident_filename)
                    
                    # Save the full frame with annotation
                    annotated_frame = frame.copy()
                    
                    # Crop the suspect with padding
                    x1, y1, x2, y2 = map(int, bbox)
                    x1_crop = max(0, x1 - crop_padding)
                    y1_crop = max(0, y1 - crop_padding)
                    x2_crop = min(frame.shape[1], x2 + crop_padding)
                    y2_crop = min(frame.shape[0], y2 + crop_padding)
                    suspect_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                    
                    # Save both the annotated full frame and the cropped suspect
                    cv2.imwrite(incident_image_path, annotated_frame)
                    crop_filename = os.path.join(screenshots_dir, f"crop_{person_id}_{int(time.time())}.jpg")
                    cv2.imwrite(crop_filename, suspect_crop)
                    
                    # Create detection record
                    from database import add_detection, add_incident
                    
                    # Prepare detection data
                    behavior_desc = "not looking at hands" if not looking_at_chest else "looking at hands"
                    detection_data = {
                        "video_id": None,  # Live feed doesn't have video ID
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "frame_number": frame_count,
                        "detection_type": "theft",
                        "confidence": 0.8,  # Default confidence
                        "bbox": bbox,
                        "class_name": "person",
                        "image_path": crop_filename,
                        "detection_metadata": {
                            "person_id": person_id,
                            "zone_type": zone_type,
                            "behavior": behavior_desc
                        }
                    }
                    
                    # Add detection to database
                    detection_id = await add_detection(detection_data)
                    
                    # Prepare incident description
                    description = f"Suspicious hand movement in {zone_type} area while {behavior_desc}"
                    
                    # Create incident record
                    incident_data = {
                        "type": "theft",
                        "timestamp": timestamp,
                        "location": f"Camera {camera_id}" if camera_id else "Unknown",
                        "description": description,
                        "image_path": crop_filename,
                        "video_url": video_path,
                        "severity": "high",
                        "confidence": 0.8,
                        "detection_ids": [detection_id],
                        "is_resolved": False
                    }
                    
                    # Create new incident
                    incident_id = await add_incident(incident_data)
                    
                    return incident_id
                    
                except Exception as e:
                    logger.error(f"Error saving theft incident: {str(e)}")
                    return None
            
            # Process frames
            frame_count = 0
            theft_detections = []
            device = 0 if torch.cuda.is_available() else 'cpu'
            current_time = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate current time in video
                current_time = frame_count / fps
                
                # Process every N frames for performance
                if frame_count % skip_frames == 0:
                    # Detect people using pose model
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose_model(frame_rgb, conf=0.5, device=device)
                    
                    # Process each detection
                    tracked_people = []
                    
                    for result in pose_results:
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            keypoints = result.keypoints.xy.cpu().numpy()
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            
                            for i in range(len(boxes)):
                                bbox = boxes[i].tolist()
                                kpts = keypoints[i].tolist() if i < len(keypoints) else []
                                
                                # Simple ID assignment - in a real system, you'd use a more robust matching method
                                person_id = f"person_{next_person_id}"
                                next_person_id += 1
                                
                                # Add to tracked people
                                tracked_people.append({
                                    "id": person_id,
                                    "bbox": bbox,
                                    "keypoints": kpts,
                                    "confidence": float(confidences[i])
                                })
                    
                    # Process each tracked person
                    for person in tracked_people:
                        person_id = person["id"]
                        keypoints = person["keypoints"]
                        bbox = person["bbox"]
                        
                        # Skip if not enough keypoints for full analysis
                        if len(keypoints) < 17:
                            continue
                        
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
                        draw_skeleton(frame, keypoints)
                        
                        # Get wrist and nose positions
                        left_wrist = keypoints[LEFT_WRIST] if LEFT_WRIST < len(keypoints) and keypoints[LEFT_WRIST][0] > 0 else None
                        right_wrist = keypoints[RIGHT_WRIST] if RIGHT_WRIST < len(keypoints) and keypoints[RIGHT_WRIST][0] > 0 else None
                        nose = keypoints[NOSE] if NOSE < len(keypoints) and keypoints[NOSE][0] > 0 else None
                        
                        # Approximate eye gaze
                        is_looking_at_chest = approximate_eye_gaze(nose, chest_box)
                        
                        # Track hands in chest/waist regions
                        for wrist, label in [(left_wrist, f"{person_id}_left"), (right_wrist, f"{person_id}_right")]:
                            if wrist is None:
                                continue
                            
                            wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                            wrist_box = [wrist_x - 10, wrist_y - 10, wrist_x + 10, wrist_y + 10]
                            cv2.rectangle(frame, (wrist_box[0], wrist_box[1]), (wrist_box[2], wrist_box[3]), (0, 255, 0), 2)
                            
                            # Check if hand is in a suspicious zone
                            in_chest = is_intersecting(wrist_box, chest_box)
                            in_left_waist = is_intersecting(wrist_box, left_waist_box)
                            in_right_waist = is_intersecting(wrist_box, right_waist_box)
                            
                            # Suspicious behavior detection
                            if in_chest or in_left_waist or in_right_waist:
                                # Determine which zone and appropriate threshold
                                zone_type = "chest" if in_chest else "waist"
                                hand_stay_time = hand_stay_time_chest if in_chest else hand_stay_time_waist
                                
                                # Start or continue timing hand position
                                if label not in hand_timers:
                                    hand_timers[label] = current_time
                                elif current_time - hand_timers[label] > hand_stay_time:
                                    # If hand has been in zone for threshold time and person isn't already flagged
                                    if person_id not in detected_theft_persons:
                                        # Increased suspicion if not looking at chest
                                        suspicion_text = "⚠️ Shoplifter!"
                                        cv2.putText(frame, suspicion_text, (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                        
                                        # Mark this person as detected
                                        detected_theft_persons.add(person_id)
                                        
                                        # Log detection
                                        duration = current_time - hand_timers[label]
                                        logger.info(f"Theft detection: Person {person_id}, hand in {zone_type} for {duration:.1f}s, looking: {is_looking_at_chest}")
                                        
                                        # Save incident to database
                                        await save_theft_incident(
                                            person_id, frame, bbox, keypoints, zone_type, is_looking_at_chest
                                        )
                                        
                                        # Record for function return
                                        theft_detection = {
                                            "type": "theft",
                                            "person_id": person_id,
                                            "confidence": 0.8,
                                            "bbox": bbox,
                                            "zone": zone_type,
                                            "frame_number": frame_count,
                                            "looking_at_chest": is_looking_at_chest,
                                            "duration": duration
                                        }
                                        theft_detections.append(theft_detection)
                                
                                # Store previous hand positions
                                previous_hand_positions[label] = (in_chest, in_left_waist, in_right_waist)
                            else:
                                # Hand not in suspicious zone - reset timer
                                if label in hand_timers:
                                    del hand_timers[label]
                        
                        # Visualization - red box for detected theft, green otherwise
                        if person_id in detected_theft_persons:
                            # Detected theft - red box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"ID: {person_id} - THEFT", (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Normal tracking - green box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Clean up timers for hands no longer tracked
                    all_current_labels = set(f"{person['id']}_{side}" for person in tracked_people for side in ["left", "right"])
                    for label in list(hand_timers.keys()):
                        if label not in all_current_labels:
                            del hand_timers[label]
                
                # Write processed frame
                out.write(frame)
                
                frame_count += 1
                
                # Report progress periodically
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Theft detection progress: {progress:.1f}%")
            
            # Release resources
            cap.release()
            out.release()
            
            return {
                "output_path": output_path,
                "screenshots_dir": screenshots_dir,
                "detections": theft_detections,
                "total_frames": frame_count,
                "total_theft_incidents": len(detected_theft_persons)
            }
            
        except Exception as e:
            logger.error(f"Error in theft detection: {str(e)}")
            raise
video_processor = VideoProcessor()