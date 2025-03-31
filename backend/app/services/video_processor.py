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
video_processor = VideoProcessor()