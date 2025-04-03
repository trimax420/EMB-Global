import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import cv2
import numpy as np
import torch
import asyncio
import logging
import time
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Set
import threading
import queue

from app.core.config import settings
from app.core.websocket import websocket_manager
from database import (
    add_detection,
    add_incident,
    update_video_status,
    add_customer_data,
    add_detections_bulk
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store active detection streams
active_detections = {}

# Video processing queue
video_queue = queue.Queue()
processing_threads = []

class Detector:
    def __init__(self):
        self.face_model = None
        self.pose_model = None
        self.object_model = None
        self.init_models()
        
        # For tracking active streams
        self.active_streams = {}
        
        # Start worker threads for video processing
        self.start_worker_threads()
    
    def init_models(self):
        """Initialize AI models for detection"""
        try:
            # First check if YOLO is available
            from ultralytics import YOLO
            from ultralytics.nn.tasks import DetectionModel
            import torch.serialization
            from torch.nn.modules.container import Sequential
            from torch.nn.modules import Conv2d, BatchNorm2d, SiLU, Module, ModuleList
            
            # Add required safe globals for PyTorch 2.6+
            torch.serialization.add_safe_globals([
                DetectionModel,
                Sequential,
                Conv2d,
                BatchNorm2d,
                SiLU,
                Module,
                ModuleList
            ])
            
            # Load face detection model
            if settings.FACE_MODEL_PATH.exists():
                self.face_model = YOLO(str(settings.FACE_MODEL_PATH))
                logger.info("Face detection model loaded successfully")
            else:
                logger.warning(f"Face model not found at {settings.FACE_MODEL_PATH}")
            
            # Load pose detection model
            if settings.POSE_MODEL_PATH.exists():
                self.pose_model = YOLO(str(settings.POSE_MODEL_PATH))
                logger.info("Pose detection model loaded successfully")
            else:
                logger.warning(f"Pose model not found at {settings.POSE_MODEL_PATH}")
            
            # Load object detection model
            if settings.OBJECT_MODEL_PATH.exists():
                self.object_model = YOLO(str(settings.OBJECT_MODEL_PATH))
                logger.info("Object detection model loaded successfully")
            else:
                logger.warning(f"Object model not found at {settings.OBJECT_MODEL_PATH}")
                
        except ImportError:
            logger.warning("YOLO package not available. Models not loaded.")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    def start_worker_threads(self):
        """Start worker threads for video processing"""
        MAX_WORKERS = 2  # Adjust based on system capability
        for _ in range(MAX_WORKERS):
            thread = threading.Thread(target=self.video_processor_worker, daemon=True)
            thread.start()
            processing_threads.append(thread)
        logger.info(f"Started {MAX_WORKERS} video processing worker threads")
    
    def video_processor_worker(self):
        """Worker function to process videos from the queue"""
        while True:
            try:
                task = video_queue.get(timeout=1)
                if task is None:  # Poison pill to stop the thread
                    break
                
                video_path, video_id, detection_type = task
                self.process_video(video_path, video_id, detection_type)
                video_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in video processor: {str(e)}")
                video_queue.task_done()
    
    def process_video(self, video_path: str, video_id: int, detection_type: str):
        """Process video and store detections"""
        try:
            # Update video status to processing
            asyncio.run(update_video_status(video_id, "processing"))
            
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            output_path = settings.PROCESSED_DIR / f"processed_{video_id}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Select appropriate model based on detection type
            if detection_type == "theft":
                model = self.object_model
                target_classes = ["person", "backpack", "handbag", "suitcase"]
            elif detection_type == "loitering":
                model = self.pose_model
            else:  # face_detection
                model = self.face_model
            
            frame_number = 0
            detections = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame for performance
                if frame_number % 5 == 0 and model:
                    # Run detection
                    results = model(frame)
                    
                    for result in results:
                        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, 'boxes') else []
                        confidences = result.boxes.conf.cpu().numpy() if hasattr(result, 'boxes') else []
                        class_ids = result.boxes.cls.cpu().numpy() if hasattr(result, 'boxes') else []
                        
                        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                            # Filter detections based on type
                            if detection_type == "theft":
                                class_name = model.names[int(cls_id)]
                                if class_name not in target_classes:
                                    continue
                            
                            # Save detection frame
                            frame_path = settings.FRAMES_DIR / f"frame_{video_id}_{frame_number}_{i}.jpg"
                            cv2.imwrite(str(frame_path), frame)
                            
                            # Store detection data
                            detection_data = {
                                "video_id": video_id,
                                "timestamp": datetime.now(),
                                "frame_number": frame_number,
                                "detection_type": detection_type,
                                "confidence": float(conf),
                                "bbox": box.tolist(),
                                "class_name": model.names[int(cls_id)] if hasattr(model, 'names') else None,
                                "image_path": str(frame_path),
                                "detection_metadata": {
                                    "fps": fps,
                                    "frame_count": frame_count,
                                    "resolution": f"{width}x{height}"
                                }
                            }
                            detections.append(detection_data)
                            
                            # Draw detection on frame
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{detection_type} {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Write processed frame
                out.write(frame)
                frame_number += 1
            
            cap.release()
            out.release()
            
            # Generate thumbnail
            thumbnail_path = settings.THUMBNAILS_DIR / f"thumb_{video_id}.jpg"
            cv2.imwrite(str(thumbnail_path), frame)
            
            # Store detections in database
            for detection in detections:
                asyncio.run(add_detection(detection))
            
            # Update video status
            asyncio.run(update_video_status(video_id, "completed", str(output_path)))
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            asyncio.run(update_video_status(video_id, "failed"))
    
    async def process_frame(self, frame: np.ndarray, detection_type: str = "all") -> List[Dict]:
        """Process a single frame and return detections using CUDA acceleration"""
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
                # Process for face detection with CUDA
                face_results = self.face_model(frame_rgb, conf=0.5, device=0 if torch.cuda.is_available() else 'cpu')
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
                # Process for object detection with CUDA
                object_results = self.object_model(frame_rgb, conf=0.5, device=0 if torch.cuda.is_available() else 'cpu')
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
    
    async def process_video_with_customer_data(self, video_path: str, video_id: int, detection_type: str, output_path: str):
        """Process video with real-time detection updates"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame and get detections
                detections = await self.process_frame(frame, detection_type)
                
                # Draw detections on frame
                for detection in detections:
                    x1, y1, x2, y2 = map(int, detection["bbox"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{detection['class_name']} {detection['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save detection frames
                if detections:
                    frame_path = settings.FRAMES_DIR / f"video_{video_id}_frame_{frame_number}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Store detections in database
                    for detection in detections:
                        detection_data = {
                            "video_id": video_id,
                            "timestamp": datetime.now(),
                            "frame_number": frame_number,
                            "detection_type": detection["type"],
                            "confidence": detection["confidence"],
                            "bbox": detection["bbox"],
                            "class_name": detection["class_name"],
                            "image_path": str(frame_path)
                        }
                        await add_detection(detection_data)
                    
                    # Broadcast detection update
                    await websocket_manager.broadcast({
                        "type": "detection_update",
                        "video_id": video_id,
                        "frame_number": frame_number,
                        "detections": detections,
                        "frame_url": f"/frames/{frame_path.name}"
                    })
                
                # Write processed frame
                out.write(frame)
                frame_number += 1
                
                # Update processing progress
                progress = (frame_number / frame_count) * 100
                await websocket_manager.broadcast({
                    "type": "processing_progress",
                    "video_id": video_id,
                    "progress": progress
                })
            
            cap.release()
            out.release()
            
            # Update video status
            await update_video_status(video_id, "completed", output_path)
            
            # Broadcast completion notification
            await websocket_manager.broadcast({
                "type": "processing_completed",
                "video_id": video_id,
                "total_detections": len(detections),
                "output_path": output_path
            })
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            await update_video_status(video_id, "failed")
            raise
    
    async def start_live_stream(self, camera_id: int, video_path: str):
        """Start processing and streaming a live video feed"""
        try:
            logger.info(f"Starting live stream for camera {camera_id} from {video_path}")
            
            # Create a task for the stream processing
            task = asyncio.create_task(self.process_live_stream(camera_id, video_path))
            
            # Store the task
            self.active_streams[camera_id] = {
                "task": task,
                "video_path": video_path,
                "start_time": datetime.now()
            }
            
            return {"status": "success", "message": f"Live stream started for camera {camera_id}"}
        
        except Exception as e:
            logger.error(f"Error starting live stream: {str(e)}")
            raise
    
    async def process_live_stream(self, camera_id: int, video_path: str):
        """Process live video stream and send frames through WebSocket"""
        try:
            logger.info(f"Opening video file: {video_path}")
            
            # Check if the file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file does not exist: {video_path}")
                raise FileNotFoundError(f"Video file not found at path: {video_path}")
            
            # Open the video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                raise ValueError(f"Failed to open video feed at path: {video_path}")
            
            # Get video properties for logging
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Successfully opened video: {video_path} ({width}x{height}, {fps} FPS, {frame_count} frames)")
            
            # Initialize detection storage for this camera
            active_detections[camera_id] = []
            
            # Set target FPS for streaming
            target_fps = min(10, fps if fps > 0 else 30)  # Don't exceed video's FPS, use 30 as fallback
            frame_interval = 1.0 / target_fps
            
            # Use a frame counter to handle looping properly
            frame_counter = 0
            frame_mod = max(1, int(fps / target_fps))  # Process every Nth frame for performance
            
            try:
                while True:
                    # Check if stream task is cancelled
                    if asyncio.current_task().cancelled():
                        break
                    
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        # Loop back for video files
                        logger.info(f"End of video reached for camera {camera_id}, looping back")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_counter = 0
                        # Add a small delay to prevent CPU thrashing on empty files
                        await asyncio.sleep(0.5)
                        continue
                    
                    # Only process every Nth frame for efficiency
                    frame_counter += 1
                    if frame_counter % frame_mod != 0:
                        await asyncio.sleep(0.01)  # Small sleep to prevent CPU thrashing
                        continue
                    
                    # Process frame for detections
                    detections = await self.process_frame(frame)
                    
                    # Update active detections
                    active_detections[camera_id] = detections
                    
                    # Draw detections on frame
                    annotated_frame = frame.copy()
                    for det in detections:
                        if "bbox" in det:
                            x1, y1, x2, y2 = map(int, det["bbox"])
                            confidence = det["confidence"]
                            class_name = det["class_name"]
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label with confidence
                            label = f"{class_name}: {confidence:.2f}"
                            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            y1 = max(y1, label_size[1])
                            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - baseline),
                                        (x1 + label_size[0], y1), (0, 255, 0), cv2.FILLED)
                            cv2.putText(annotated_frame, label, (x1, y1 - baseline),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    # Add frame counter to the image
                    frame_info = f"Frame: {frame_counter}/{frame_count if frame_count > 0 else 'unknown'}"
                    cv2.putText(annotated_frame, frame_info, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Convert frame to base64 for streaming
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Calculate progress for video files
                    progress = None
                    if frame_count > 0:
                        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        progress = (current_frame / frame_count) * 100
                        
                    # Send frame and detections through WebSocket
                    await websocket_manager.broadcast({
                        "type": "live_detection",
                        "camera_id": camera_id,
                        "frame": frame_base64,
                        "detections": [
                            {
                                "type": det["type"],
                                "class_name": det["class_name"],
                                "confidence": float(det["confidence"]),
                                "bbox": det["bbox"]
                            }
                            for det in detections
                        ],
                        "timestamp": datetime.now().isoformat(),
                        "progress": progress,
                        "frame_number": frame_counter,
                        "total_frames": frame_count if frame_count > 0 else None
                    })
                    
                    # Store detections in database periodically
                    if len(detections) > 0 and frame_counter % 100 == 0:  # Every 100 processed frames
                        # Save annotated frame
                        frame_path = settings.FRAMES_DIR / f"camera_{camera_id}_frame_{int(time.time())}.jpg"
                        cv2.imwrite(str(frame_path), annotated_frame)
                        
                        # Store detections
                        bulk_detections = []
                        for det in detections:
                            detection_data = {
                                "camera_id": camera_id,
                                "timestamp": datetime.now(),
                                "frame_number": frame_counter,
                                "detection_type": det["type"],
                                "confidence": det["confidence"],
                                "bbox": det["bbox"],
                                "class_name": det["class_name"],
                                "image_path": str(frame_path)
                            }
                            bulk_detections.append(detection_data)
                        
                        if bulk_detections:
                            await add_detections_bulk(bulk_detections)
                    
                    # Maintain target FPS
                    await asyncio.sleep(frame_interval)
                    
            except asyncio.CancelledError:
                logger.info(f"Live stream processing for camera {camera_id} was cancelled")
            finally:
                cap.release()
                # Clean up
                if camera_id in active_detections:
                    del active_detections[camera_id]
        
        except Exception as e:
            logger.error(f"Error in process_live_stream: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Clean up
            if camera_id in active_detections:
                del active_detections[camera_id]
    
    async def stop_live_stream(self, camera_id: int):
        """Stop a live video stream"""
        if camera_id in self.active_streams:
            try:
                # Cancel the task
                self.active_streams[camera_id]["task"].cancel()
                # Wait for it to complete
                try:
                    await self.active_streams[camera_id]["task"]
                except asyncio.CancelledError:
                    pass
                
                # Remove from active streams
                del self.active_streams[camera_id]
                
                return {"status": "success", "message": f"Live stream stopped for camera {camera_id}"}
            
            except Exception as e:
                logger.error(f"Error stopping live stream: {str(e)}")
                raise
        else:
            return {"status": "error", "message": f"No active stream for camera {camera_id}"}

# Create a global detector instance
detector = Detector()
