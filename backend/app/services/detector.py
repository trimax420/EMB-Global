import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DetectionService:
    """Service for person detection in video frames using YOLOv5"""
    
    def __init__(self, model_path: str = "yolov5s.pt", conf_threshold: float = 0.5):
        """
        Initialize the detection service with YOLOv5 model
        
        Args:
            model_path: Path to the YOLOv5 model weights
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load YOLOv5 model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            self.model.to(self.device)
            self.model.conf = conf_threshold
            self.model.classes = [0]  # Only detect people (class 0 in COCO)
            logger.info(f"Detection model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
            # Fallback to OpenCV's HOG detector if YOLO fails
            logger.info("Falling back to OpenCV HOG detector")
            self.model = None
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons in a frame
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: (x1, y1, x2, y2) coordinates
            - confidence: Detection confidence
            - class_id: Always 0 for person
        """
        start_time = time.time()
        detections = []
        
        if self.model is not None:
            # Use YOLOv5 for detection
            results = self.model(frame)
            
            # Process detections
            for pred in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2, conf, class_id = pred
                if conf >= self.conf_threshold and int(class_id) == 0:  # Person class
                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": float(conf),
                        "class_id": 0
                    })
        else:
            # Fallback to HOG detector
            boxes, weights = self.hog.detectMultiScale(
                frame, 
                winStride=(8, 8),
                padding=(4, 4), 
                scale=1.05
            )
            
            for i, (x, y, w, h) in enumerate(boxes):
                detections.append({
                    "bbox": (x, y, x + w, y + h),
                    "confidence": float(weights[i]),
                    "class_id": 0
                })
        
        processing_time = time.time() - start_time
        logger.debug(f"Detection completed in {processing_time:.4f}s. Found {len(detections)} persons.")
        
        return detections
    
    def classify_persons(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify detected persons (staff vs customer, gender estimation)
        
        Args:
            frame: Input frame
            detections: List of person detections
            
        Returns:
            Enhanced detections with added classification information
        """
        # This is a simplified implementation
        # In a real system, you would use a trained classifier here
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            person_img = frame[y1:y2, x1:x2]
            
            # Simple color-based heuristic for staff (assuming staff wears uniforms)
            # This is just a placeholder - would need real model training
            if person_img.size > 0:
                hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
                # Example: staff wears blue uniforms
                blue_lower = np.array([100, 50, 50])
                blue_upper = np.array([130, 255, 255])
                blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
                blue_ratio = cv2.countNonZero(blue_mask) / (person_img.shape[0] * person_img.shape[1])
                
                detection["is_staff"] = blue_ratio > 0.3
                
                # Placeholder for gender estimation
                # In a real system, use a proper classifier
                detection["gender"] = "unknown"
            
        return detections