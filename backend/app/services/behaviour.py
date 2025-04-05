import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Any, Tuple
import time
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class BehaviorAnalysisService:
    """Service for analyzing person behavior for security surveillance"""
    
    def __init__(self, zone_definitions: List[Dict] = None):
        """
        Initialize behavior analysis service
        
        Args:
            zone_definitions: List of zone dictionaries with 'id', 'name', and 'coordinates' keys
        """
        self.zones = zone_definitions or []
        
        # Initialize MediaPipe Pose for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Track person dwell times
        self.person_dwell = {}  # track_id -> {zone_id -> dwell_time}
        self.last_seen = {}  # track_id -> timestamp
        
        # Track suspicious poses
        self.suspicious_activity = {}  # track_id -> suspicious_activity_count
        
        logger.info("Behavior analysis service initialized")
    
    def analyze_frame(self, frame: np.ndarray, tracked_persons: List[Dict[str, Any]], 
                     timestamp: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Analyze behavior in the current frame
        
        Args:
            frame: Current video frame
            tracked_persons: List of tracked person dictionaries
            timestamp: Frame timestamp
            
        Returns:
            Tuple of (analytics_data, alerts)
        """
        analytics_data = []
        alerts = []
        
        # Update zone assignments
        zone_counts = {zone['id']: 0 for zone in self.zones}
        
        for person in tracked_persons:
            track_id = person.get('track_id')
            if track_id is None:
                continue
                
            # Initialize tracking data for new persons
            if track_id not in self.person_dwell:
                self.person_dwell[track_id] = {zone['id']: 0 for zone in self.zones}
                self.last_seen[track_id] = timestamp
                self.suspicious_activity[track_id] = 0
            
            # Get person's current zone
            bbox = person['bbox']
            current_zone = self._get_person_zone(bbox)
            person['zone_id'] = current_zone
            
            if current_zone:
                zone_counts[current_zone] += 1
            
            # Update dwell time in zones
            time_delta = timestamp - self.last_seen[track_id]
            if current_zone:
                self.person_dwell[track_id][current_zone] += time_delta
            
            # Check for suspicious poses
            pose_result = self._analyze_pose(frame, bbox)
            
            if pose_result['suspicious']:
                self.suspicious_activity[track_id] += 1
                
                # Generate alert if suspicious activity threshold reached
                if self.suspicious_activity[track_id] >= 3:  # Alert after 3 suspicious frames
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'alert_type': 'suspicious_pose',
                        'severity': 3,
                        'zone_id': current_zone,
                        'track_id': track_id,
                        'description': f"Suspicious posture detected: {pose_result['details']}",
                        'bbox': bbox
                    }
                    alerts.append(alert)
                    # Reset counter after alert
                    self.suspicious_activity[track_id] = 0
            
            # Check for loitering (long dwell time in a zone)
            if current_zone and self.person_dwell[track_id][current_zone] > 300:  # 5 minutes threshold
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'alert_type': 'loitering',
                    'severity': 2,
                    'zone_id': current_zone,
                    'track_id': track_id,
                    'description': f"Person loitering in zone {current_zone} for over 5 minutes",
                    'bbox': bbox
                }
                alerts.append(alert)
                # Reset dwell time after alert
                self.person_dwell[track_id][current_zone] = 0
            
            # Update last seen timestamp
            self.last_seen[track_id] = timestamp
        
        # Generate zone analytics
        for zone in self.zones:
            zone_id = zone['id']
            analytics = {
                'zone_id': zone_id,
                'timestamp': datetime.now().isoformat(),
                'person_count': zone_counts[zone_id],
                'avg_dwell_time': self._calculate_avg_dwell_time(zone_id)
            }
            analytics_data.append(analytics)
        
        # Clean up tracking data for persons not seen recently
        self._cleanup_tracking_data(timestamp)
        
        return analytics_data, alerts
    
    def _calculate_avg_dwell_time(self, zone_id: int) -> float:
        """
        Calculate average dwell time for a zone across all tracked persons
        
        Args:
            zone_id: Zone ID
            
        Returns:
            Average dwell time in seconds
        """
        zone_dwell_times = [
            person_dwell[zone_id] 
            for person_dwell in self.person_dwell.values() 
            if zone_id in person_dwell
        ]
        
        if not zone_dwell_times:
            return 0.0
            
        return sum(zone_dwell_times) / len(zone_dwell_times)
    
    def _cleanup_tracking_data(self, current_timestamp: float) -> None:
        """
        Remove tracking data for persons not seen recently
        
        Args:
            current_timestamp: Current frame timestamp
        """
        stale_tracks = []
        
        for track_id, last_seen_time in self.last_seen.items():
            if current_timestamp - last_seen_time > 60:  # 1 minute timeout
                stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            if track_id in self.person_dwell:
                del self.person_dwell[track_id]
            if track_id in self.last_seen:
                del self.last_seen[track_id]
            if track_id in self.suspicious_activity:
                del self.suspicious_activity[track_id]
    
    def _get_person_zone(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Determine which zone a person is in based on bounding box
        
        Args:
            bbox: Person bounding box (x1, y1, x2, y2)
            
        Returns:
            Zone ID or None if person is not in any defined zone
        """
        person_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        for zone in self.zones:
            polygon = np.array(zone['coordinates'], np.int32)
            if cv2.pointPolygonTest(polygon, person_center, False) >= 0:
                return zone['id']
        
        return None
    
    def _analyze_pose(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Analyze person pose for suspicious behavior
        
        Args:
            frame: Current video frame
            bbox: Person bounding box
            
        Returns:
            Dictionary with pose analysis results
        """
        x1, y1, x2, y2 = bbox
        
        # Extract person ROI
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return {'suspicious': False, 'details': 'Invalid ROI'}
        
        # Convert to RGB for MediaPipe
        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        
        # Get pose landmarks
        results = self.pose.process(rgb_roi)
        
        if not results.pose_landmarks:
            return {'suspicious': False, 'details': 'No pose detected'}
        
        landmarks = results.pose_landmarks.landmark
        
        # Detect suspicious pose patterns
        # Example: bent over posture that could indicate shoplifting
        # This is a simplified implementation - real system would need more sophisticated detection
        
        # Check if hands are near pockets
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Check for hand-to-pocket proximity
        left_hand_near_pocket = self._calculate_distance(left_wrist, left_hip) < 0.15
        right_hand_near_pocket = self._calculate_distance(right_wrist, right_hip) < 0.15
        
        # Check for bent over posture
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        bent_over = (nose.y > left_hip.y) or (nose.y > right_hip.y)
        
        if (left_hand_near_pocket or right_hand_near_pocket) and bent_over:
            return {
                'suspicious': True,
                'details': 'Possible shoplifting posture detected'
            }
        elif left_hand_near_pocket or right_hand_near_pocket:
            return {
                'suspicious': True,
                'details': 'Hand near pocket detected'
            }
        elif bent_over:
            return {
                'suspicious': True,
                'details': 'Bent over posture detected'
            }
        
        return {'suspicious': False, 'details': 'Normal posture'}
    
    def _calculate_distance(self, landmark1, landmark2) -> float:
        """Calculate Euclidean distance between two landmarks"""
        return np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)