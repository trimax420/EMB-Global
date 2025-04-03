import os
import json
import time
import logging
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from datetime import datetime

import cv2
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from aiortc.mediastreams import VideoFrame

from app.core.config import settings

logger = logging.getLogger(__name__)

# WebRTC peer connections storage
pcs = set()
# Media relay for sharing a single video source
relay = MediaRelay()

class MLVideoStream(VideoStreamTrack):
    """
    A video stream track that processes frames with ML models before sending.
    This allows us to stream inference results in real-time via WebRTC.
    """
    def __init__(self, processor, video_path=None, detection_type="all", camera_id=None, resolution="original"):
        super().__init__()
        self.processor = processor
        self.frame_counter = 0
        self.camera_id = camera_id
        self.detection_type = detection_type
        self.resolution = resolution
        self.video_path = video_path
        
        # Set up logging
        logger.info(f"Initializing MLVideoStream with video_path={video_path}, camera_id={camera_id}, resolution={resolution}")
        
        # Open video source (file or camera)
        if video_path and os.path.exists(video_path):
            logger.info(f"Opening video file: {video_path}")
            self.capture = cv2.VideoCapture(video_path)
        elif video_path:
            # Try to find the file in uploads/raw directory
            upload_dir = os.path.join(settings.UPLOAD_DIR, "raw")
            basename = os.path.basename(video_path)
            potential_path = os.path.join(upload_dir, basename)
            
            if os.path.exists(potential_path):
                logger.info(f"Found video at: {potential_path}")
                self.video_path = potential_path
                self.capture = cv2.VideoCapture(potential_path)
            else:
                logger.warning(f"Video not found, using default test video")
                # Use a default test video or pattern
                self.capture = cv2.VideoCapture(0)  # Change to a test pattern or default video
        else:
            # Use camera
            logger.info(f"Opening camera id: {camera_id if camera_id is not None else 0}")
            self.capture = cv2.VideoCapture(camera_id if camera_id is not None else 0)
        
        # Check if video/camera opened successfully
        if not self.capture.isOpened():
            logger.error(f"Failed to open video source: {video_path or camera_id}")
            # Don't raise exception, we'll handle this in recv by returning a black frame
            self.width = 1280
            self.height = 720
        else:
            # Get original dimensions
            self.original_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Original video dimensions: {self.original_width}x{self.original_height}")
            
            # Set target dimensions based on resolution parameter
            if resolution == "original":
                self.width = self.original_width
                self.height = self.original_height
            elif resolution == "720p":
                self.height = 720
                self.width = int((self.height / self.original_height) * self.original_width)
                self.width = self.width - (self.width % 2)  # Ensure even width
            elif resolution == "1080p":
                self.height = 1080
                self.width = int((self.height / self.original_height) * self.original_width)
                self.width = self.width - (self.width % 2)  # Ensure even width
            else:
                # Default to 720p if invalid resolution specified
                self.height = 720
                self.width = 1280
        
        # Get FPS
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.fps = 30  # Default FPS if not available
        
        logger.info(f"Stream initialized with resolution: {self.width}x{self.height}, FPS: {self.fps}")

    async def recv(self):
        try:
            self.frame_counter += 1
            
            # Read frame from video/camera
            ret, frame = self.capture.read()
            
            if not ret:
                # If end of video, loop back to beginning
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.capture.read()
                if not ret:
                    # If still can't read, use a black frame
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    cv2.putText(frame, "Video Error", (50, self.height//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Resize frame to target resolution if needed
            frame_height, frame_width = frame.shape[:2]
            if self.width != frame_width or self.height != frame_height:
                try:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    logger.debug(f"Resized frame from {frame_width}x{frame_height} to {self.width}x{self.height}")
                except Exception as e:
                    logger.error(f"Error resizing frame: {str(e)}")
                    # Continue with original frame if resize fails
            
            # Create a copy for processing
            processed_frame = frame.copy()
            
            # Add overlay text with resolution and frame number
            cv2.putText(processed_frame, f"Frame: {self.frame_counter}", (10, 30), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Resolution: {self.width}x{self.height}", (10, 60), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert the processed frame to the format expected by aiortc
            video_frame = VideoFrame.from_ndarray(processed_frame, format="bgr24")
            video_frame.pts = self.frame_counter * 1000  # timestamp in milliseconds
            video_frame.time_base = 1000  # milliseconds timebase
            
            return video_frame
            
        except Exception as e:
            logger.error(f"Error in recv method: {str(e)}")
            logger.exception("Full stack trace:")
            
            # Return a black frame with error message as fallback
            dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, f"Error: {str(e)[:30]}...", (50, self.height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            video_frame = VideoFrame.from_ndarray(dummy_frame, format="bgr24")
            video_frame.pts = self.frame_counter * 1000  # timestamp in milliseconds
            video_frame.time_base = 1000  # milliseconds timebase
            return video_frame
    
    async def process_loitering_detection(self, frame, detections):
        """Process loitering detection logic in real-time"""
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        person_detections = [d for d in detections if d.get("class_name") == "person"]
        
        # Extract deep features for person tracking
        current_persons = []
        for det in person_detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                continue
                
            # Extract person region
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue
                
            # Use simple color histogram as feature (you can replace with more sophisticated features)
            try:
                # Simple histogram feature
                hist_feature = cv2.calcHist([person_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist_feature = cv2.normalize(hist_feature, hist_feature).flatten()
                
                # Store the person with bbox and features
                current_persons.append({
                    "bbox": bbox,
                    "features": hist_feature,
                    "detection_conf": det.get("confidence", 0),
                    "id": -1  # Will be assigned during matching
                })
            except Exception as e:
                logger.error(f"Error extracting features: {str(e)}")
        
        # Match current persons with tracked persons
        if not self.tracked_persons:
            # Initialize tracking on first frame
            for i, person in enumerate(current_persons):
                person["id"] = i
                self.tracked_persons.append(person)
                self.loitering_timers[i] = 0
        else:
            # Match persons across frames
            for cur_person in current_persons:
                best_match = -1
                best_score = float('inf')
                best_iou = 0
                
                for i, tracked in enumerate(self.tracked_persons):
                    # Calculate IoU between bounding boxes
                    iou = self.calculate_iou(cur_person["bbox"], tracked["bbox"])
                    
                    # Calculate feature distance
                    feat_dist = np.linalg.norm(cur_person["features"] - tracked["features"])
                    
                    # Combined score (lower is better)
                    score = feat_dist * (1.0 - iou)
                    
                    if iou > 0.3 and score < best_score:
                        best_match = i
                        best_score = score
                        best_iou = iou
                
                if best_match >= 0:
                    # Update tracked person with new position and add to loitering time
                    cur_person["id"] = self.tracked_persons[best_match]["id"]
                    person_id = cur_person["id"]
                    
                    # Update loitering time (assuming ~30fps)
                    if person_id in self.loitering_timers:
                        self.loitering_timers[person_id] += time_delta if 'time_delta' in locals() else 0.033
                    else:
                        self.loitering_timers[person_id] = 0
                    
                    # Replace the tracked person with the current one
                    self.tracked_persons[best_match] = cur_person
                else:
                    # New person detected
                    person_id = max([p["id"] for p in self.tracked_persons], default=-1) + 1
                    cur_person["id"] = person_id
                    self.tracked_persons.append(cur_person)
                    self.loitering_timers[person_id] = 0
        
        # Remove stale tracks (not seen in current frame)
        current_ids = [p["id"] for p in current_persons]
        self.tracked_persons = [p for p in self.tracked_persons if p["id"] in current_ids]
        
        # Draw loitering visualization
        for person in self.tracked_persons:
            person_id = person["id"]
            x1, y1, x2, y2 = person["bbox"]
            loitering_time = self.loitering_timers.get(person_id, 0)
            
            if loitering_time >= self.loitering_threshold:
                # Person is loitering
                color = (0, 0, 255)  # Red for loitering
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add loitering label with time
                label = f"LOITERING ID:{person_id} ({loitering_time:.1f}s)"
                cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add alert box at top of screen
                alert_text = f"⚠️ LOITERING ALERT: Person {person_id} ({loitering_time:.1f}s)"
                cv2.putText(vis_frame, alert_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Regular tracking visualization
                color = (0, 255, 0)  # Green for normal tracking
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add tracking label with time
                label = f"Person {person_id} ({loitering_time:.1f}s)"
                cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    async def process_theft_detection(self, frame, detections):
        """Process theft detection logic in real-time"""
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Get person and object detections
        person_detections = [d for d in detections if d.get("class_name") == "person"]
        object_detections = [d for d in detections if d.get("class_name") not in ["person", "background"]]
        
        # Get pose keypoints for hand tracking
        keypoint_detections = [d for d in detections if "keypoints" in d and d.get("keypoints")]
        
        # Track objects first
        if not self.tracked_objects and object_detections:
            # Initialize object tracking
            for i, obj in enumerate(object_detections):
                obj_bbox = obj.get("bbox", [])
                if len(obj_bbox) != 4:
                    continue
                
                self.tracked_objects.append({
                    "id": i,
                    "bbox": obj_bbox,
                    "class_name": obj.get("class_name", "object"),
                    "confidence": obj.get("confidence", 0)
                })
        else:
            # Update tracked objects
            for obj in object_detections:
                obj_bbox = obj.get("bbox", [])
                if len(obj_bbox) != 4:
                    continue
                
                # Find best match in tracked objects
                best_match = -1
                best_iou = 0
                
                for i, tracked_obj in enumerate(self.tracked_objects):
                    iou = self.calculate_iou(obj_bbox, tracked_obj["bbox"])
                    if iou > 0.5 and iou > best_iou:
                        best_match = i
                        best_iou = iou
                
                if best_match >= 0:
                    # Update tracked object
                    self.tracked_objects[best_match]["bbox"] = obj_bbox
                    self.tracked_objects[best_match]["confidence"] = obj.get("confidence", 0)
                else:
                    # New object
                    obj_id = max([o["id"] for o in self.tracked_objects], default=-1) + 1
                    self.tracked_objects.append({
                        "id": obj_id,
                        "bbox": obj_bbox,
                        "class_name": obj.get("class_name", "object"),
                        "confidence": obj.get("confidence", 0)
                    })
        
        # Process person detections with keypoints for hand tracking
        for person in keypoint_detections:
            keypoints = person.get("keypoints", [])
            if not keypoints or len(keypoints) < 17:  # COCO format has 17 keypoints
                continue
            
            person_bbox = person.get("bbox", [])
            if len(person_bbox) != 4:
                continue
            
            # Get wrist keypoints (9 and 10 in COCO format)
            left_wrist = keypoints[9] if len(keypoints) > 9 else None
            right_wrist = keypoints[10] if len(keypoints) > 10 else None
            
            # Check for hand proximity to objects
            for obj in self.tracked_objects:
                obj_bbox = obj["bbox"]
                obj_id = obj["id"]
                
                # Define hand proximity to object
                hand_near_object = False
                
                # Check left wrist
                if left_wrist and left_wrist[0] > 0 and left_wrist[1] > 0:
                    if self.is_point_near_bbox(left_wrist, obj_bbox, threshold=50):
                        hand_near_object = True
                        # Draw circle on left wrist
                        cv2.circle(vis_frame, (int(left_wrist[0]), int(left_wrist[1])), 5, (0, 0, 255), -1)
                
                # Check right wrist
                if right_wrist and right_wrist[0] > 0 and right_wrist[1] > 0:
                    if self.is_point_near_bbox(right_wrist, obj_bbox, threshold=50):
                        hand_near_object = True
                        # Draw circle on right wrist
                        cv2.circle(vis_frame, (int(right_wrist[0]), int(right_wrist[1])), 5, (0, 0, 255), -1)
                
                # Create a unique key for each person-object pair
                pair_key = f"person_{person.get('id', 0)}_obj_{obj_id}"
                
                if hand_near_object:
                    # Start or update timer
                    if pair_key not in self.hand_object_timers:
                        self.hand_object_timers[pair_key] = 0
                    else:
                        self.hand_object_timers[pair_key] += time_delta if 'time_delta' in locals() else 0.033
                    
                    # Check for theft
                    if self.hand_object_timers[pair_key] >= self.hand_stay_time:
                        # Theft detected!
                        x1, y1, x2, y2 = map(int, obj_bbox)
                        
                        # Draw theft alert on object
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        theft_text = f"⚠️ THEFT: {obj['class_name']} ({self.hand_object_timers[pair_key]:.1f}s)"
                        cv2.putText(vis_frame, theft_text, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Draw global alert
                        cv2.putText(vis_frame, f"⚠️ THEFT ALERT: {obj['class_name']}", (10, 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Connect hand to object with a line
                        if left_wrist and left_wrist[0] > 0 and left_wrist[1] > 0:
                            cv2.line(vis_frame, (int(left_wrist[0]), int(left_wrist[1])), 
                                     (int((x1+x2)/2), int((y1+y2)/2)), (0, 0, 255), 2)
                        
                        if right_wrist and right_wrist[0] > 0 and right_wrist[1] > 0:
                            cv2.line(vis_frame, (int(right_wrist[0]), int(right_wrist[1])), 
                                     (int((x1+x2)/2), int((y1+y2)/2)), (0, 0, 255), 2)
                else:
                    # Reset timer if hand is not near the object
                    if pair_key in self.hand_object_timers:
                        self.hand_object_timers[pair_key] = 0
        
        # Draw objects
        for obj in self.tracked_objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            
            # Check if this object has a theft timer
            theft_timers = [v for k, v in self.hand_object_timers.items() if f"_obj_{obj['id']}" in k]
            max_timer = max(theft_timers) if theft_timers else 0
            
            if max_timer > 0 and max_timer < self.hand_stay_time:
                # Object with hand near it but not theft yet
                color = (0, 165, 255)  # Orange for potential theft
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{obj['class_name']} - TOUCH ({max_timer:.1f}s)"
                cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif max_timer < self.hand_stay_time:
                # Regular object
                color = (0, 255, 0)  # Green for normal object
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{obj['class_name']} {obj['confidence']:.2f}"
                cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    def calculate_iou(self, box1, box2):
        """Calculate intersection over union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        if union_area <= 0:
            return 0.0
            
        return intersection_area / union_area
    
    def is_point_near_bbox(self, point, bbox, threshold=20):
        """Check if a point (keypoint) is near a bounding box"""
        x, y = point[0], point[1]
        x1, y1, x2, y2 = bbox
        
        # Check if point is inside the bbox
        if x >= x1 and x <= x2 and y >= y1 and y <= y2:
            return True
            
        # Check if point is near the bbox (within threshold)
        if (x < x1-threshold or x > x2+threshold or 
            y < y1-threshold or y > y2+threshold):
            return False
            
        return True
        
    def stop(self):
        if self.capture:
            self.capture.release()
        super().stop()

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
            
        # WebRTC video streams by camera ID
        self.video_streams = {}
        
    async def create_rtc_connection(self, offer, camera_id=None, video_path=None, detection_type="all", resolution="original"):
        """
        Create a WebRTC peer connection for streaming video with ML processing
        
        Args:
            offer: WebRTC offer from client
            camera_id: Camera ID for streaming
            video_path: Video file path for streaming
            detection_type: Type of detection to perform on frames
            resolution: Preferred resolution for the video stream (e.g. '720p', '1080p', 'original')
            
        Returns:
            RTCSessionDescription: WebRTC answer
        """
        try:
            # Create RTCPeerConnection with ICE servers
            logger.info(f"Creating RTCPeerConnection for {video_path or camera_id}")
            pc = RTCPeerConnection({
                "iceServers": [
                    {"urls": "stun:stun.l.google.com:19302"},
                    {"urls": "stun:stun1.l.google.com:19302"}
                ]
            })
            
            # Keep track of peer connections
            pcs.add(pc)
            
            # Create video stream
            logger.info(f"Creating MLVideoStream with resolution {resolution}")
            video_stream = MLVideoStream(
                processor=self,
                video_path=video_path,
                detection_type=detection_type,
                camera_id=camera_id,
                resolution=resolution
            )
            
            # Add track to peer connection
            logger.info("Adding video track to peer connection")
            pc.addTrack(relay.subscribe(video_stream))
            
            # Log ICE connection state changes
            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                logger.info(f"ICE connection state changed to: {pc.iceConnectionState}")
                if pc.iceConnectionState == "failed" or pc.iceConnectionState == "closed":
                    await self.close_rtc_connection(pc)
            
            # Log connection state changes
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"Connection state changed to: {pc.connectionState}")
                if pc.connectionState == "failed" or pc.connectionState == "closed":
                    await self.close_rtc_connection(pc)
            
            # Set remote description
            logger.info("Setting remote description (client offer)")
            await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
            
            # Create answer
            logger.info("Creating answer")
            answer = await pc.createAnswer()
            logger.info("Setting local description (answer)")
            await pc.setLocalDescription(answer)
            
            logger.info(f"WebRTC connection established successfully for {video_path or camera_id}")
            
            return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            
        except Exception as e:
            logger.error(f"Error creating WebRTC connection: {str(e)}")
            logger.exception("Full stack trace:")
            raise
    
    async def close_rtc_connection(self, pc):
        """Close a WebRTC peer connection and cleanup resources"""
        logger.info("Closing WebRTC peer connection")
        pcs.discard(pc)
        
        # Close PC
        await pc.close()
        
        # Cleanup video streams that are no longer used
        for stream_id, stream in list(self.video_streams.items()):
            in_use = False
            for other_pc in pcs:
                for sender in other_pc.getSenders():
                    if sender.track == stream:
                        in_use = True
                        break
                if in_use:
                    break
            
            if not in_use:
                logger.info(f"Stopping unused video stream: {stream_id}")
                stream.stop()
                del self.video_streams[stream_id]
        
    async def close_all_rtc_connections(self):
        """Close all WebRTC peer connections"""
        logger.info(f"Closing all {len(pcs)} WebRTC peer connections")
        coros = [pc.close() for pc in pcs]
        await asyncio.gather(*coros)
        pcs.clear()
        
        # Stop all video streams
        for stream_id, stream in self.video_streams.items():
            stream.stop()
        self.video_streams.clear()

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
                        # for detection in frame_detections:
                        #     await manager.broadcast(json.dumps({
                        #         "type": "detection",
                        #         "video_id": video_id,
                        #         "detection": detection
                        #     }))
                    
                    except Exception as detection_error:
                        logger.error(f"Detection error: {str(detection_error)}")
                
                # Write processed frame to output video
                out.write(frame)
                frame_number += 1
                
                # Periodic progress updates via WebSocket
                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    
                    # Avoid too frequent updates
                    # if progress - last_progress_update >= 5:
                    #     await manager.broadcast(json.dumps({
                    #         "type": "processing_progress",
                    #         "video_id": video_id,
                    #         "progress": progress,
                    #         "total_frames": total_frames
                    #     }))
                    #     last_progress_update = progress
            
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
            # await manager.broadcast(json.dumps({
            #     "type": "processing_completed",
            #     "video_id": video_id,
            #     "results": processing_results
            # }))
            
            return processing_results
        
        except Exception as e:
            # Handle and log any unexpected errors
            logger.error(f"Error processing video {video_id}: {str(e)}")
            
            # Broadcast error message
            # await manager.broadcast(json.dumps({
            #     "type": "processing_error",
            #     "video_id": video_id,
            #     "error": str(e)
            # }))
            
            raise

    async def process_loitering_detection_smooth(self, video_path, output_path, camera_id=None, threshold_time=10):
        """
        Process video for loitering detection with improved person re-identification and smooth tracking.
        Maintains consistent person IDs across frames to prevent duplication issues.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save processed video
            camera_id: ID of the camera feed (if available)
            threshold_time: Time threshold (in seconds) to consider loitering
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Starting smooth loitering detection on {video_path}")
            
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
            tracked_persons = {}  # {person_id: {first_seen, last_seen, position, features, accumulated_time, etc.}}
            loitering_db_entries = {}  # {person_id: {'db_id': id, 'last_updated': timestamp}}
            loitering_incidents = set()  # For avoiding duplicate detections
            next_person_id = 1
            
            # Define helper functions
            def extract_person_features(frame, bbox):
                """Extract color histogram features from person region for re-identification"""
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
            
            def match_person(features, bbox, tracked_persons, iou_threshold=0.3, feature_threshold=0.5):
                """Match a detected person with previously tracked persons using both IoU and features"""
                best_match_id = None
                best_match_score = 0
                
                # Skip matching if no features
                if features is None:
                    return None
                    
                # Calculate center point of current bbox
                curr_center_x = (bbox[0] + bbox[2]) / 2
                curr_center_y = (bbox[1] + bbox[3]) / 2
                
                for person_id, person_data in tracked_persons.items():
                    if 'last_seen_frame' not in person_data:
                        continue
                        
                    # Skip persons not seen recently (more than 30 frames ago)
                    frame_diff = frame_count - person_data['last_seen_frame']
                    if frame_diff > 30:
                        continue
                    
                    # Calculate IoU between current bbox and tracked person's bbox
                    prev_bbox = person_data['bbox']
                    iou = calculate_iou(bbox, prev_bbox)
                    
                    # Calculate distance between centers for more robustness
                    prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
                    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
                    center_distance = np.sqrt((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)
                    
                    # Calculate normalized center distance (lower is better)
                    max_frame_dim = max(width, height)
                    norm_distance = 1.0 - min(1.0, center_distance / (max_frame_dim * 0.5))
                    
                    # Calculate feature similarity if IoU is reasonable
                    feature_similarity = 0
                    if iou > 0.1 or norm_distance > 0.7:
                        if 'features' in person_data and person_data['features'] is not None:
                            feature_similarity = cv2.compareHist(
                                features.reshape(-1, 1),
                                person_data['features'].reshape(-1, 1),
                                cv2.HISTCMP_CORREL
                            )
                    
                    # Combined matching score - weight both position and appearance
                    combined_score = (iou * 0.4) + (norm_distance * 0.3) + (feature_similarity * 0.3)
                    
                    if combined_score > best_match_score and combined_score > iou_threshold:
                        best_match_score = combined_score
                        best_match_id = person_id
                
                return best_match_id
                
            def calculate_iou(box1, box2):
                """Calculate IoU between two bounding boxes"""
                # Convert to [x1, y1, x2, y2] format
                box1 = [float(x) for x in box1]
                box2 = [float(x) for x in box2]
                
                # Calculate intersection area
                x_left = max(box1[0], box2[0])
                y_top = max(box1[1], box2[1])
                x_right = min(box1[2], box2[2])
                y_bottom = min(box1[3], box2[3])
                
                if x_right < x_left or y_bottom < y_top:
                    return 0.0  # No intersection
                    
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                
                # Calculate box areas
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                
                # Calculate union area
                union_area = box1_area + box2_area - intersection_area
                
                # Calculate IoU
                iou = intersection_area / union_area if union_area > 0 else 0
                
                return iou
                
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
                    
                return [x1, y1, x2, y2]
            
            # Helper function to save loitering person to database
            async def save_loitering_person(person_id, bbox, frame, accumulated_time, camera_id):
                """Save or update loitering person in the database"""
                if person_id in loitering_incidents:
                    return None  # Already recorded
                    
                try:
                    # Create a filename for the person's image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    person_image_filename = f"loitering_{person_id}_{timestamp}.jpg"
                    person_image_path = os.path.join(screenshots_dir, person_image_filename)
                    
                    # Save the full frame with annotation
                    annotated_frame = frame.copy()
                    
                    # Draw bounding box around person
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Add text label with accumulated time
                    cv2.putText(annotated_frame, 
                            f"Loitering: {accumulated_time:.1f}s", 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Save the annotated frame
                    cv2.imwrite(person_image_path, annotated_frame)
                    
                    # Also crop and save just the person
                    person_crop = frame[y1:y2, x1:x2]
                    crop_filename = os.path.join(screenshots_dir, f"crop_{person_id}_{timestamp}.jpg")
                    cv2.imwrite(crop_filename, person_crop)
                    
                    # Format the incident data
                    incident_data = {
                        "type": "loitering",
                        "timestamp": datetime.now(),
                        "location": f"Camera {camera_id}" if camera_id else "Unknown",
                        "description": f"Person loitering for {accumulated_time:.1f} seconds",
                        "image_path": person_image_path,
                        "video_url": video_path,
                        "severity": "medium",
                        "confidence": 0.9,
                        "duration": accumulated_time,
                        "is_resolved": False,
                    }
                    
                    # Create detection record
                    from database import add_detection, add_incident
                    
                    # Prepare detection data
                    detection_data = {
                        "video_id": None,  # Live feed doesn't have a video ID
                        "camera_id": camera_id,
                        "timestamp": datetime.now(),
                        "frame_number": frame_count,
                        "detection_type": "loitering",
                        "confidence": 0.9,
                        "bbox": bbox,
                        "class_name": "person",
                        "image_path": crop_filename,
                        "detection_metadata": {
                            "person_id": person_id,
                            "accumulated_time": accumulated_time
                        }
                    }
                    
                    # Add detection to database
                    detection_id = await add_detection(detection_data)
                    
                    # Add detection ID to incident data
                    incident_data["detection_ids"] = [detection_id]
                    
                    # Create new incident
                    incident_id = await add_incident(incident_data)
                    
                    # Record database entry for future updates
                    loitering_db_entries[person_id] = {
                        "id": incident_id,
                        "last_updated": datetime.now()
                    }
                    
                    # Mark as recorded
                    loitering_incidents.add(person_id)
                    
                    logger.info(f"Created new loitering record for person {person_id}")
                    return incident_id
                    
                except Exception as e:
                    logger.error(f"Error saving loitering person: {str(e)}")
                    return None
            
            # Process frames
            frame_count = 0
            loitering_detections = []
            device = 0 if torch.cuda.is_available() else 'cpu'
            current_time = 0
            
            # For smoother visualization, carry forward detections in skipped frames
            last_processed_detections = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate current time in video
                current_time = frame_count / fps
                
                # Process frames regularly to maintain detection accuracy,
                # but maintain tracking data for every frame
                if frame_count % settings.SKIP_FRAMES == 0:
                    # Detect people
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.object_model(frame_rgb, conf=0.5, device=device)
                    
                    # Extract person detections
                    current_detections = []
                    
                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy()
                            
                            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                                class_name = self.object_model.names[int(cls_id)]
                                
                                # Only track people
                                if class_name == "person" and conf > 0.5:
                                    bbox = box.tolist()
                                    
                                    # Extract features for tracking
                                    features = extract_person_features(frame, bbox)
                                    
                                    # Match with existing tracked persons
                                    matched_id = match_person(features, bbox, tracked_persons)
                                    
                                    if matched_id is not None:
                                        # Update existing person
                                        person_id = matched_id
                                        
                                        # Calculate time since last seen
                                        if 'last_seen_time' in tracked_persons[person_id]:
                                            time_elapsed = current_time - tracked_persons[person_id]['last_seen_time']
                                            
                                            # Only accumulate time if seen recently (within 2 seconds)
                                            if time_elapsed < 2.0:
                                                tracked_persons[person_id]['accumulated_time'] += time_elapsed
                                        
                                        # Update tracking info
                                        tracked_persons[person_id].update({
                                            'bbox': bbox,
                                            'features': features,
                                            'position': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                                            'last_seen_frame': frame_count,
                                            'last_seen_time': current_time,
                                            'active': True
                                        })
                                    else:
                                        # New person detected
                                        person_id = f"person_{next_person_id}"
                                        next_person_id += 1
                                        
                                        # Initialize tracking data
                                        tracked_persons[person_id] = {
                                            'first_seen_frame': frame_count,
                                            'first_seen_time': current_time,
                                            'last_seen_frame': frame_count,
                                            'last_seen_time': current_time,
                                            'bbox': bbox,
                                            'features': features,
                                            'position': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                                            'accumulated_time': 0.0,
                                            'active': True,
                                            'loitering_detected': False,
                                            'screenshot_saved': False
                                        }
                                    
                                    # Add to current detections
                                    current_detections.append({
                                        'id': person_id,
                                        'bbox': bbox,
                                        'confidence': float(conf)
                                    })
                    
                    # Update last processed detections for smoother visualization
                    last_processed_detections = current_detections
                else:
                    # Use previously processed detections for visualization
                    current_detections = last_processed_detections
                
                # Process all current detections for visualization and loitering detection
                for detection in current_detections:
                    person_id = detection['id']
                    bbox = detection['bbox']
                    
                    if person_id not in tracked_persons:
                        continue  # Skip if tracking data missing
                    
                    # Get person data
                    person_data = tracked_persons[person_id]
                    accumulated_time = person_data.get('accumulated_time', 0.0)
                    
                    # Check for loitering based on accumulated time
                    loitering_detected = accumulated_time > threshold_time
                    
                    # Update loitering status
                    person_data['loitering_detected'] = loitering_detected
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw visualization
                    if loitering_detected:
                        # Draw red box for loitering
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{person_id}: {accumulated_time:.1f}s", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Save screenshot and database entry if not already done
                        if not person_data.get('screenshot_saved', False) and person_id not in loitering_incidents:
                            # Save to database
                            await save_loitering_person(person_id, bbox, frame, accumulated_time, camera_id)
                            
                            # Record loitering detection for return value
                            loitering_detection = {
                                "type": "loitering",
                                "person_id": person_id,
                                "confidence": 0.9,
                                "bbox": bbox,
                                "time_present": accumulated_time,
                                "frame_number": frame_count
                            }
                            loitering_detections.append(loitering_detection)
                            
                            # Mark as saved
                            person_data['screenshot_saved'] = True
                            logger.info(f"Loitering detected for {person_id}, time: {accumulated_time:.1f}s")
                    else:
                        # Draw green box for normal tracking
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{person_id}: {accumulated_time:.1f}s", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Clean up old tracked persons to prevent memory issues
                if frame_count % 100 == 0:
                    current_ids = set(detection['id'] for detection in current_detections)
                    for person_id in list(tracked_persons.keys()):
                        # Remove if not seen in the last 60 frames (2 seconds at 30fps)
                        if person_id not in current_ids and \
                        'last_seen_frame' in tracked_persons[person_id] and \
                        frame_count - tracked_persons[person_id]['last_seen_frame'] > 60:
                            del tracked_persons[person_id]
                
                # Write processed frame
                out.write(frame)
                
                frame_count += 1
                
                # Report progress periodically
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Loitering detection progress: {progress:.1f}% (tracked persons: {len(tracked_persons)})")
            
            # Final database updates for all loitering people
            try:
                for person_id, person_data in tracked_persons.items():
                    if person_data.get('loitering_detected', False) and not person_data.get('screenshot_saved', False):
                        # Get the last known position
                        bbox = person_data.get('bbox')
                        if bbox:
                            # Create a frame if we have a stored one, otherwise use the last frame
                            await save_loitering_person(
                                person_id,
                                bbox,
                                frame,  # Use the last frame
                                person_data['accumulated_time'],
                                camera_id
                            )
                            
                            # Record for return value
                            loitering_detection = {
                                "type": "loitering",
                                "person_id": person_id,
                                "confidence": 0.9,
                                "bbox": bbox,
                                "time_present": person_data['accumulated_time'],
                                "frame_number": frame_count - 1
                            }
                            loitering_detections.append(loitering_detection)
            except Exception as e:
                logger.error(f"Error in final database updates: {str(e)}")
            
            # Release resources
            cap.release()
            out.release()
            
            # Generate summary statistics
            loitering_persons = [
                {"person_id": p_id, "time": data['accumulated_time']}
                for p_id, data in tracked_persons.items()
                if data.get('loitering_detected', False)
            ]
            
            return {
                "output_path": output_path,
                "screenshots_dir": screenshots_dir,
                "detections": loitering_detections,
                "total_frames": frame_count,
                "unique_persons": len(tracked_persons),
                "loitering_persons": loitering_persons,
                "loitering_count": len(loitering_persons)
            }
            
        except Exception as e:
            logger.error(f"Error in loitering detection: {str(e)}")
            raise

    async def process_theft_detection_smooth(self, video_path, output_path, screenshots_dir, hand_stay_time=2.0, camera_id=None):
        """
        Process video for theft detection with persistent tracking to ensure smooth inference.
        Prevents duplication of person IDs and maintains consistent tracking across frames.
        
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
            logger.info(f"Starting smooth theft detection on {video_path}")
            
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
            
            # Persistent tracking variables for smooth inference
            # Track people across frames using their position and appearance
            tracked_persons = {}  # {person_id: {last_seen, bbox, keypoints, features, etc.}}
            hand_positions = {}   # {person_id_hand: [(time, position, zone),...]}
            detected_theft_incidents = set()  # Track already detected incidents
            next_person_id = 1  # Counter for new person IDs
            
            # Helper functions
            def extract_person_features(frame, bbox):
                """Extract color histogram features from person region for re-identification"""
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
            
            def match_person(features, bbox, tracked_persons, iou_threshold=0.3, feature_threshold=0.5):
                """Match a detected person with previously tracked persons using both IoU and features"""
                best_match_id = None
                best_match_score = 0
                
                # Skip matching if no features
                if features is None:
                    return None
                    
                # Calculate center point of current bbox
                curr_center_x = (bbox[0] + bbox[2]) / 2
                curr_center_y = (bbox[1] + bbox[3]) / 2
                
                for person_id, person_data in tracked_persons.items():
                    if 'last_seen_frame' not in person_data:
                        continue
                        
                    # Skip persons not seen recently (more than 30 frames ago)
                    frame_diff = frame_count - person_data['last_seen_frame']
                    if frame_diff > 30:
                        continue
                    
                    # Calculate IoU between current bbox and tracked person's bbox
                    prev_bbox = person_data['bbox']
                    iou = calculate_iou(bbox, prev_bbox)
                    
                    # Calculate distance between centers for more robustness
                    prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
                    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
                    center_distance = np.sqrt((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)
                    
                    # Calculate normalized center distance (lower is better)
                    max_frame_dim = max(width, height)
                    norm_distance = 1.0 - min(1.0, center_distance / (max_frame_dim * 0.5))
                    
                    # Calculate feature similarity if IoU is reasonable
                    feature_similarity = 0
                    if iou > 0.1 or norm_distance > 0.7:
                        if 'features' in person_data and person_data['features'] is not None:
                            feature_similarity = cv2.compareHist(
                                features.reshape(-1, 1),
                                person_data['features'].reshape(-1, 1),
                                cv2.HISTCMP_CORREL
                            )
                    
                    # Combined matching score - weight both position and appearance
                    combined_score = (iou * 0.4) + (norm_distance * 0.3) + (feature_similarity * 0.3)
                    
                    if combined_score > best_match_score and combined_score > iou_threshold:
                        best_match_score = combined_score
                        best_match_id = person_id
                
                return best_match_id
                
            def calculate_iou(box1, box2):
                """Calculate IoU between two bounding boxes"""
                # Convert to [x1, y1, x2, y2] format
                box1 = [float(x) for x in box1]
                box2 = [float(x) for x in box2]
                
                # Calculate intersection area
                x_left = max(box1[0], box2[0])
                y_top = max(box1[1], box2[1])
                x_right = min(box1[2], box2[2])
                y_bottom = min(box1[3], box2[3])
                
                if x_right < x_left or y_bottom < y_top:
                    return 0.0  # No intersection
                    
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                
                # Calculate box areas
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                
                # Calculate union area
                union_area = box1_area + box2_area - intersection_area
                
                # Calculate IoU
                iou = intersection_area / union_area if union_area > 0 else 0
                
                return iou
                
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
                    
                    # Create a copy of the frame for annotation
                    annotated_frame = frame.copy()
                    
                    # Draw bounding box and skeleton on the annotated frame
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    draw_skeleton(annotated_frame, keypoints, color=(255, 0, 0))
                    
                    # Add text label
                    behavior_desc = "not looking at hands" if not looking_at_chest else "looking at hands"
                    cv2.putText(annotated_frame, 
                            f"THEFT: Hand in {zone_type}, {behavior_desc}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Crop the suspect with padding
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
                    logger.info(f"Created theft incident with ID {incident_id} for person {person_id}")
                    
                    return incident_id
                    
                except Exception as e:
                    logger.error(f"Error saving theft incident: {str(e)}")
                    return None
            
            # Process frames
            frame_count = 0
            theft_detections = []
            device = 0 if torch.cuda.is_available() else 'cpu'
            current_time = 0
            
            # For smoother visualization, carry forward detections in skipped frames
            last_processed_detections = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate current time in video
                current_time = frame_count / fps
                
                # Process frames regularly to maintain detection accuracy,
                # but draw visualizations on every frame for smooth output
                if frame_count % skip_frames == 0:
                    # Detect people using pose model
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose_model(frame_rgb, conf=0.5, device=device)
                    
                    # Process each detection and maintain consistent IDs
                    current_detections = []
                    
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
                                features = extract_person_features(frame, bbox)
                                
                                # Match with existing tracked persons
                                matched_id = match_person(features, bbox, tracked_persons)
                                
                                if matched_id is not None:
                                    # Update existing person
                                    person_id = matched_id
                                    # Update tracking info
                                    tracked_persons[person_id].update({
                                        'bbox': bbox,
                                        'keypoints': kpts,
                                        'features': features,
                                        'last_seen_frame': frame_count,
                                        'last_seen_time': current_time
                                    })
                                else:
                                    # New person detected
                                    person_id = f"person_{next_person_id}"
                                    next_person_id += 1
                                    tracked_persons[person_id] = {
                                        'bbox': bbox,
                                        'keypoints': kpts,
                                        'features': features,
                                        'first_seen_frame': frame_count,
                                        'first_seen_time': current_time,
                                        'last_seen_frame': frame_count,
                                        'last_seen_time': current_time
                                    }
                                
                                # Add to current detections
                                current_detections.append({
                                    'id': person_id,
                                    'bbox': bbox,
                                    'keypoints': kpts,
                                    'confidence': float(confidences[i])
                                })
                    
                    # Update last processed detections
                    last_processed_detections = current_detections
                else:
                    # Use previously processed detections but update positions if possible
                    # This is a simplified version - in a production system, you might use a 
                    # motion model to predict positions between processed frames
                    current_detections = last_processed_detections
                
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
                    draw_skeleton(frame, keypoints)
                    
                    # Get wrist and nose positions
                    left_wrist = keypoints[LEFT_WRIST] if LEFT_WRIST < len(keypoints) and keypoints[LEFT_WRIST][0] > 0 else None
                    right_wrist = keypoints[RIGHT_WRIST] if RIGHT_WRIST < len(keypoints) and keypoints[RIGHT_WRIST][0] > 0 else None
                    nose = keypoints[NOSE] if NOSE < len(keypoints) and keypoints[NOSE][0] > 0 else None
                    
                    # Approximate eye gaze
                    is_looking_at_chest = approximate_eye_gaze(nose, chest_box)
                    
                    # Track hands in chest/waist regions
                    for wrist, hand_label in [(left_wrist, f"{person_id}_left"), (right_wrist, f"{person_id}_right")]:
                        if wrist is None:
                            continue
                        
                        wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                        wrist_box = [wrist_x - 10, wrist_y - 10, wrist_x + 10, wrist_y + 10]
                        cv2.rectangle(frame, (wrist_box[0], wrist_box[1]), (wrist_box[2], wrist_box[3]), (0, 255, 0), 2)
                        
                        # Check if hand is in a suspicious zone
                        in_chest = is_intersecting(wrist_box, chest_box)
                        in_left_waist = is_intersecting(wrist_box, left_waist_box)
                        in_right_waist = is_intersecting(wrist_box, right_waist_box)
                        
                        # Initialize tracking for this hand if not already tracked
                        if hand_label not in hand_positions:
                            hand_positions[hand_label] = []
                        
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
                        hand_positions[hand_label].append((current_time, (wrist_x, wrist_y), zone))
                        
                        # Limit history size to avoid memory issues
                        if len(hand_positions[hand_label]) > 60:  # Assuming 30fps, keep 2 seconds of history
                            hand_positions[hand_label].pop(0)
                        
                        # Analyze hand positions for theft detection
                        if in_chest or in_left_waist or in_right_waist:
                            # Determine which zone and appropriate threshold
                            zone_type = "chest" if in_chest else "waist"
                            threshold = hand_stay_time_chest if in_chest else hand_stay_time_waist
                            
                            # Count how long the hand has been in this zone continuously
                            zone_duration = 0
                            continuous_in_zone = True
                            
                            # Check hand position history in reverse order
                            for idx in range(len(hand_positions[hand_label]) - 1, 0, -1):
                                past_time, _, past_zone = hand_positions[hand_label][idx]
                                prev_time, _, prev_zone = hand_positions[hand_label][idx - 1]
                                
                                # If zone changes, break the continuity
                                if past_zone != zone and past_zone != "other":
                                    continuous_in_zone = False
                                    break
                                
                                # If zone matches, add to duration
                                if past_zone == zone:
                                    zone_duration += (past_time - prev_time)
                            
                            # Check if duration exceeds threshold
                            if zone_duration >= threshold:
                                # Generate a unique incident ID combining person and zone
                                incident_id = f"{person_id}_{zone}"
                                
                                # Check if this incident has already been detected
                                if incident_id not in detected_theft_incidents:
                                    # Mark as detected
                                    detected_theft_incidents.add(incident_id)
                                    
                                    # Log detection
                                    logger.info(f"Theft detection: Person {person_id}, hand in {zone} for {zone_duration:.1f}s, looking: {is_looking_at_chest}")
                                    
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
                                        "duration": zone_duration
                                    }
                                    theft_detections.append(theft_detection)
                    
                    # Visualization - red box for detected theft, green otherwise
                    incident_id_chest = f"{person_id}_chest"
                    incident_id_waist = f"{person_id}_waist"
                    
                    if incident_id_chest in detected_theft_incidents or incident_id_waist in detected_theft_incidents:
                        # Detected theft - red box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"ID: {person_id} - THEFT", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # Normal tracking - green box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Clean up old tracked persons to prevent memory issues
                if frame_count % 100 == 0:
                    current_ids = set(detection['id'] for detection in current_detections)
                    for person_id in list(tracked_persons.keys()):
                        # Remove if not seen in the last 60 frames (2 seconds at 30fps)
                        if person_id not in current_ids and \
                        'last_seen_frame' in tracked_persons[person_id] and \
                        frame_count - tracked_persons[person_id]['last_seen_frame'] > 60:
                            del tracked_persons[person_id]
                    
                    # Also clean up hand positions for persons no longer tracked
                    valid_persons = set(tracked_persons.keys())
                    for hand_label in list(hand_positions.keys()):
                        person_id = hand_label.split('_')[0]
                        if person_id not in valid_persons:
                            del hand_positions[hand_label]
                
                # Write processed frame
                out.write(frame)
                
                frame_count += 1
                
                # Report progress periodically
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Theft detection progress: {progress:.1f}% (tracked persons: {len(tracked_persons)})")
            
            # Release resources
            cap.release()
            out.release()
            
            return {
                "output_path": output_path,
                "screenshots_dir": screenshots_dir,
                "detections": theft_detections,
                "total_frames": frame_count,
                "total_theft_incidents": len(detected_theft_incidents),
                "unique_persons_tracked": len(tracked_persons)
            }
            
        except Exception as e:
            logger.error(f"Error in theft detection: {str(e)}")
            raise

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

    # Add this method to track people by face
    async def track_person_by_face(self, face_image_path, video_path, output_path=None, 
                                screenshot_dir=None, similarity_threshold=0.6, skip_frames=5):
        """
        Track a person in a video based on their face.
        
        Args:
            face_image_path: Path to face image to track
            video_path: Path to video to search in
            output_path: Path for output video with tracking visualization
            screenshot_dir: Directory to save detection screenshots
            similarity_threshold: Threshold for face matching (0-1)
            skip_frames: Process every Nth frame
            
        Returns:
            Dictionary with tracking results
        """
        try:
            # Generate output paths if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"tracked_{timestamp}_{Path(video_path).name}"
                output_path = os.path.join(settings.PROCESSED_DIR, output_filename)
            
            if screenshot_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_dir = os.path.join(settings.SCREENSHOTS_DIR, f"person_tracking_{timestamp}")
                os.makedirs(screenshot_dir, exist_ok=True)
            
            # Use the face matching method
            results = await self.process_face_matching(
                face_image_path, 
                video_path, 
                output_path, 
                screenshot_dir,
                similarity_threshold,
                skip_frames
            )
            
            # Create tracking results file
            tracking_job_id = f"tracking_{uuid.uuid4().hex[:8]}"
            results_path = os.path.join(settings.PROCESSED_DIR, f"{tracking_job_id}_results.pkl")
            
            # Format the results
            tracking_results = {
                "status": "completed",
                "message": "Person tracking completed",
                "progress": 100,
                "detections": results["detections"],
                "frames_processed": results["total_frames"],
                "total_matches": results["total_matches"],
                "output_path": results["output_path"],
                "screenshots": results.get("screenshots", [])
            }
            
            # Save results
            with open(results_path, 'wb') as f:
                pickle.dump(tracking_results, f)
            
            # Add detailed results
            results["tracking_job_id"] = tracking_job_id
            results["results_path"] = results_path
            
            return results
            
        except Exception as e:
            logger.error(f"Error tracking person by face: {str(e)}")
            raise

    # Add a method to store customer face encodings
    async def add_customer_face_encoding(self, customer_id, face_image_path):
        """
        Extract face encoding from an image and store it for a customer.
        
        Args:
            customer_id: ID of the customer to update
            face_image_path: Path to the customer's face image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the face image
            face_image = face_recognition.load_image_file(face_image_path)
            face_locations = face_recognition.face_locations(face_image)
            
            if not face_locations:
                logger.error(f"No face detected in the image: {face_image_path}")
                return False
            
            # Extract face encoding
            face_encoding = face_recognition.face_encodings(face_image, [face_locations[0]])[0]
            
            # Save to database
            success = await update_customer_face_encoding(customer_id, face_encoding.tolist())
            
            if success:
                logger.info(f"Updated face encoding for customer {customer_id}")
            else:
                logger.error(f"Failed to update face encoding for customer {customer_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding customer face encoding: {str(e)}")
            return False

video_processor = VideoProcessor()