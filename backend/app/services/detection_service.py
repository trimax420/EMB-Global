from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
import time
import uuid
import cv2
import numpy as np
from datetime import datetime

from database import add_detection, add_incident
from app.core.config import settings
from app.core.websocket import websocket_manager

# Set up logging
logger = logging.getLogger(__name__)

# Track active model inference tasks
active_inference_tasks = {}

# Store active detection streams and their models
active_streams = {}

class DetectionModel:
    """Base class for detection models"""
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_loaded = False
        self.model = None
    
    async def load(self):
        """Load model - to be implemented by subclasses"""
        self.is_loaded = True
        return True
    
    async def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on a frame - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement predict method")
    
    async def unload(self):
        """Unload model from memory"""
        self.model = None
        self.is_loaded = False


class ObjectDetectionModel(DetectionModel):
    """Object Detection Model implementation"""
    async def load(self):
        """Load a mock object detection model for demo purposes"""
        logger.info(f"Loading object detection model: {self.model_name}")
        # In a real app, you would load your model here
        # self.model = load_your_model_here()
        await asyncio.sleep(1)  # Simulate loading time
        self.is_loaded = True
        return True

    async def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run object detection inference"""
        if not self.is_loaded:
            await self.load()
        
        # In a real app, you would run inference here
        # results = self.model.predict(frame)
        
        # For demo, simulate some random detections
        height, width = frame.shape[:2]
        detections = []
        
        # Random number of detections (0-5)
        num_detections = np.random.randint(0, 6)
        
        classes = ["person", "car", "bus", "bottle", "chair", "laptop"]
        
        for _ in range(num_detections):
            # Random detection parameters
            class_idx = np.random.randint(0, len(classes))
            class_name = classes[class_idx]
            confidence = np.random.uniform(0.6, 0.95)
            
            # Random bounding box
            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(0, height - 100)
            w = np.random.randint(50, min(200, width - x1))
            h = np.random.randint(50, min(200, height - y1))
            
            detections.append({
                "type": "object",
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x1 + w, y1 + h],
                "timestamp": datetime.now().isoformat()
            })
        
        return detections


class TheftDetectionModel(DetectionModel):
    """Theft Detection Model implementation"""
    async def load(self):
        """Load theft detection model"""
        logger.info(f"Loading theft detection model: {self.model_name}")
        # In a real app, you would load your model here
        await asyncio.sleep(1)  # Simulate loading time
        self.is_loaded = True
        return True

    async def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run theft detection inference"""
        if not self.is_loaded:
            await self.load()
        
        # In a real app, you would run inference here
        # For demo, simulate occasional theft detections
        height, width = frame.shape[:2]
        detections = []
        
        # 5% chance of theft detection
        if np.random.random() < 0.05:
            # Random position for the detection
            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(0, height - 100)
            w = np.random.randint(50, min(200, width - x1))
            h = np.random.randint(50, min(200, height - y1))
            
            detections.append({
                "type": "theft",
                "confidence": np.random.uniform(0.7, 0.95),
                "bbox": [x1, y1, x1 + w, y1 + h],
                "zone": np.random.choice(["chest", "waist"]),
                "timestamp": datetime.now().isoformat()
            })
            
            # Create incident for theft detection
            incident_id = str(uuid.uuid4())
            
            # Log the incident
            logger.info(f"Theft detected! Creating incident {incident_id}")
        
        return detections


class LoiteringDetectionModel(DetectionModel):
    """Loitering Detection Model implementation"""
    async def load(self):
        """Load loitering detection model"""
        logger.info(f"Loading loitering detection model: {self.model_name}")
        # In a real app, you would load your model here
        await asyncio.sleep(1)  # Simulate loading time
        self.is_loaded = True
        return True

    async def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run loitering detection inference"""
        if not self.is_loaded:
            await self.load()
        
        # In a real app, you would run inference here
        # For demo, simulate occasional loitering detections
        height, width = frame.shape[:2]
        detections = []
        
        # 3% chance of loitering detection
        if np.random.random() < 0.03:
            # Random position for the detection
            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(0, height - 100)
            w = np.random.randint(50, min(200, width - x1))
            h = np.random.randint(50, min(200, height - y1))
            
            time_present = np.random.uniform(10, 30)  # seconds
            
            detections.append({
                "type": "loitering",
                "confidence": np.random.uniform(0.7, 0.95),
                "bbox": [x1, y1, x1 + w, y1 + h],
                "time_present": time_present,
                "timestamp": datetime.now().isoformat()
            })
            
            # Create incident for long loitering
            if time_present > 20:
                incident_id = str(uuid.uuid4())
                logger.info(f"Loitering detected! Creating incident {incident_id}")
        
        return detections


async def run_detection_pipeline(
    camera_id: int,
    frame: np.ndarray,
    models: List[DetectionModel]
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Run multiple detection models on a single frame
    
    Args:
        camera_id: Camera identifier
        frame: The video frame to process
        models: List of detection models to run
        
    Returns:
        Tuple of (list of detections, annotated frame)
    """
    all_detections = []
    annotated_frame = frame.copy()
    
    # Process frame with each model
    for model in models:
        model_detections = await model.predict(frame)
        
        # Add model detections to results
        for detection in model_detections:
            # Add camera ID
            detection["camera_id"] = camera_id
            all_detections.append(detection)
            
            # Draw bounding box on annotated frame
            if "bbox" in detection:
                bbox = detection["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Set color based on detection type
                color = (0, 255, 0)  # Default green
                if detection["type"] == "theft":
                    color = (0, 0, 255)  # Red for theft
                elif detection["type"] == "loitering":
                    color = (255, 165, 0)  # Orange for loitering
                
                # Draw bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                conf_text = f"{detection.get('confidence', 1.0)*100:.1f}%"
                label = f"{detection.get('class_name', detection['type'])}: {conf_text}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Store detections in database
    for detection in all_detections:
        # Clone detection dict to avoid modifying the original
        db_detection = detection.copy()
        
        # Convert datetime to string if needed
        if "timestamp" in db_detection and isinstance(db_detection["timestamp"], str):
            db_detection["timestamp"] = datetime.fromisoformat(db_detection["timestamp"])
        
        try:
            # Store in database asynchronously
            # Avoid blocking the main detection pipeline
            asyncio.create_task(store_detection(db_detection))
        except Exception as e:
            logger.error(f"Error storing detection: {e}")
    
    return all_detections, annotated_frame


async def store_detection(detection: Dict[str, Any]):
    """Store detection in the database"""
    try:
        # Transform detection to match database schema
        detection_record = {
            "camera_id": detection.get("camera_id"),
            "frame_number": detection.get("frame_number", 0),
            "timestamp": detection.get("timestamp", datetime.now()),
            "detection_type": detection.get("type", "unknown"),
            "confidence": detection.get("confidence", 1.0),
            "bbox": detection.get("bbox"),
            "class_name": detection.get("class_name", detection.get("type")),
            "keypoints": detection.get("keypoints"),
            "zone": detection.get("zone"),
            "detection_metadata": {
                k: v for k, v in detection.items() 
                if k not in ["camera_id", "frame_number", "timestamp", "detection_type", 
                           "confidence", "bbox", "class_name", "keypoints", "zone"]
            }
        }
        
        # Save to database
        detection_id = await add_detection(detection_record)
        
        # For theft and loitering, create incidents
        if detection["type"] in ["theft", "loitering"]:
            incident_data = {
                "timestamp": detection_record["timestamp"],
                "type": detection_record["detection_type"],
                "location": f"Camera {detection_record['camera_id']}",
                "severity": "high" if detection["type"] == "theft" else "medium",
                "description": f"{detection['type'].capitalize()} detected",
                "detection_ids": [detection_id],
                "confidence": detection_record["confidence"]
            }
            
            # Add incident to database
            await add_incident(incident_data)
    
    except Exception as e:
        logger.error(f"Error storing detection in database: {e}")


async def start_model_inference(
    camera_id: int,
    stream_source: str,
    detection_types: List[str] = None
) -> Dict[str, Any]:
    """
    Start model inference on a camera stream
    
    Args:
        camera_id: Camera identifier
        stream_source: Path or URL to video stream
        detection_types: List of detection types to enable
        
    Returns:
        Dict with status information
    """
    # Default to all detection types if none specified
    if not detection_types:
        detection_types = ["object", "theft", "loitering"]
    
    # Initialize models based on requested detection types
    models = []
    for detection_type in detection_types:
        if detection_type == "object":
            models.append(ObjectDetectionModel("yolov5"))
        elif detection_type == "theft":
            models.append(TheftDetectionModel("theft_detector"))
        elif detection_type == "loitering":
            models.append(LoiteringDetectionModel("loitering_detector"))
    
    # Check if this camera already has an active inference task
    if camera_id in active_inference_tasks:
        # Cancel existing task
        active_inference_tasks[camera_id].cancel()
    
    # Create a task to run inference in the background
    task = asyncio.create_task(
        run_inference_loop(camera_id, stream_source, models)
    )
    
    # Store the task
    active_inference_tasks[camera_id] = task
    
    # Store active stream info
    active_streams[camera_id] = {
        "camera_id": camera_id,
        "stream_source": stream_source,
        "detection_types": detection_types,
        "models": models,
        "start_time": datetime.now(),
        "frame_count": 0
    }
    
    return {
        "status": "success",
        "message": f"Started model inference on camera {camera_id}",
        "camera_id": camera_id,
        "detection_types": detection_types,
        "start_time": datetime.now().isoformat()
    }


async def run_inference_loop(
    camera_id: int,
    stream_source: str,
    models: List[DetectionModel]
) -> None:
    """
    Run continuous inference on a video stream
    
    Args:
        camera_id: Camera identifier
        stream_source: Path or URL to video stream
        models: List of detection models to run
    """
    # Initialize video capture
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        logger.error(f"Failed to open video stream: {stream_source}")
        return
    
    logger.info(f"Started inference loop for camera {camera_id} with {len(models)} models")
    
    try:
        frame_count = 0
        
        while True:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                # If video file ended, loop back to beginning
                if stream_source.startswith(('http://', 'https://', 'rtsp://')):
                    logger.warning(f"Stream {stream_source} ended or error occurred. Reconnecting...")
                    # Wait a moment before reconnecting
                    await asyncio.sleep(2)
                    # Try to reconnect
                    cap.release()
                    cap = cv2.VideoCapture(stream_source)
                    if not cap.isOpened():
                        logger.error(f"Failed to reconnect to stream: {stream_source}")
                        break
                    continue
                else:
                    # For local video files, reset to beginning
                    logger.info(f"Restarting video file: {stream_source}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            
            # Process frame (only every 5 frames to reduce load)
            if frame_count % 5 == 0:
                # Run detection pipeline
                detections, annotated_frame = await run_detection_pipeline(
                    camera_id, frame, models
                )
                
                # Update stream stats
                if camera_id in active_streams:
                    active_streams[camera_id]["frame_count"] = frame_count
                
                # Encode the frame for sending over WebSocket
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                encoded_frame = buffer.tobytes()
                
                # Send results and frame to WebSocket clients
                await websocket_manager.broadcast({
                    "type": "live_detection",
                    "camera_id": camera_id,
                    "frame": encoded_frame,
                    "detections": detections,
                    "frame_number": frame_count
                })
            
            # Increment frame counter
            frame_count += 1
            
            # Add a small delay to prevent CPU overload
            await asyncio.sleep(0.01)
    
    except asyncio.CancelledError:
        logger.info(f"Inference loop for camera {camera_id} was cancelled")
    except Exception as e:
        logger.error(f"Error in inference loop for camera {camera_id}: {e}")
    finally:
        # Clean up
        cap.release()
        # Remove from active streams
        if camera_id in active_streams:
            del active_streams[camera_id]
        # Remove from active tasks
        if camera_id in active_inference_tasks:
            del active_inference_tasks[camera_id]


async def stop_model_inference(camera_id: int) -> Dict[str, Any]:
    """
    Stop model inference on a camera stream
    
    Args:
        camera_id: Camera identifier
    
    Returns:
        Dict with status information
    """
    if camera_id in active_inference_tasks:
        # Cancel the inference task
        active_inference_tasks[camera_id].cancel()
        # Wait for task to finish
        try:
            await asyncio.wait_for(active_inference_tasks[camera_id], timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        
        # Clean up
        del active_inference_tasks[camera_id]
        
        if camera_id in active_streams:
            # Get stats before removing
            stats = {
                "frame_count": active_streams[camera_id]["frame_count"],
                "duration": (datetime.now() - active_streams[camera_id]["start_time"]).total_seconds()
            }
            # Remove from active streams
            del active_streams[camera_id]
            
            return {
                "status": "success", 
                "message": f"Stopped model inference on camera {camera_id}",
                "camera_id": camera_id,
                "stats": stats
            }
    
    return {
        "status": "error",
        "message": f"No active inference for camera {camera_id}",
        "camera_id": camera_id
    }


def get_active_inferences() -> Dict[int, Dict[str, Any]]:
    """Get all active inference processes"""
    return {
        camera_id: {
            "camera_id": info["camera_id"],
            "stream_source": info["stream_source"],
            "detection_types": info["detection_types"],
            "start_time": info["start_time"].isoformat(),
            "frame_count": info["frame_count"],
            "active_models": [model.model_name for model in info["models"]]
        }
        for camera_id, info in active_streams.items()
    } 