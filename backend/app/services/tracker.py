import numpy as np
from typing import Dict, List, Any
import cv2
from scipy.optimize import linear_sum_assignment

class PersonTracker:
    """
    Simple IOU-based tracker for person tracking across frames
    For a production system, consider using DeepSORT or ByteTrack
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """
        Initialize tracker
        
        Args:
            iou_threshold: IOU threshold for track association
            max_age: Maximum number of frames to keep a track alive without detection
        """
        self.tracks = {}  # track_id -> track_info
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age
    
    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detection dictionaries with bbox
            frame: Current frame (optional, for visualization)
            
        Returns:
            List of active tracks with track_id
        """
        # If no tracks yet, initialize them
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {
                    "bbox": det["bbox"],
                    "age": 0,
                    "time_visible": 1,
                    "last_detection": det,
                    "is_staff": det.get("is_staff", False),
                    "gender": det.get("gender", "unknown")
                }
                det["track_id"] = self.next_id
                self.next_id += 1
            return detections
        
        # Compute IOU between current detections and existing tracks
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for i, det in enumerate(detections):
            for j, (track_id, track) in enumerate(self.tracks.items()):
                iou_matrix[i, j] = self._compute_iou(det["bbox"], track["bbox"])
        
        # Use Hungarian algorithm for optimal assignment
        det_indices, track_indices = linear_sum_assignment(-iou_matrix)
        
        # Update matched tracks
        matched_det_indices = set()
        matched_track_ids = set()
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                track_id = list(self.tracks.keys())[track_idx]
                matched_track_ids.add(track_id)
                matched_det_indices.add(det_idx)
                
                # Update track info
                self.tracks[track_id]["bbox"] = detections[det_idx]["bbox"]
                self.tracks[track_id]["age"] = 0
                self.tracks[track_id]["time_visible"] += 1
                self.tracks[track_id]["last_detection"] = detections[det_idx]
                
                # Add track_id to detection
                detections[det_idx]["track_id"] = track_id
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                self.tracks[self.next_id] = {
                    "bbox": det["bbox"],
                    "age": 0,
                    "time_visible": 1,
                    "last_detection": det,
                    "is_staff": det.get("is_staff", False),
                    "gender": det.get("gender", "unknown")
                }
                det["track_id"] = self.next_id
                self.next_id += 1
        
        # Update age of unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                self.tracks[track_id]["age"] += 1
                
                # Remove old tracks
                if self.tracks[track_id]["age"] > self.max_age:
                    del self.tracks[track_id]
        
        return detections
    
    def _compute_iou(self, bbox1, bbox2) -> float:
        """
        Compute Intersection over Union between two bounding boxes
        
        Args:
            bbox1, bbox2: Bounding boxes in format (x1, y1, x2, y2)
            
        Returns:
            IOU score between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if there is an intersection
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        # Compute areas
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Compute IOU
        union_area = bbox1_area + bbox2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou