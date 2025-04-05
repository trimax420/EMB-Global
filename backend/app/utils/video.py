import cv2
import os
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
from sqlalchemy.orm import Session

from app.models.detection import DetectionEvent
from app.models.alert import Alert
from app.models.analytics import ZoneAnalytics, AnalyticsAggregates
from app.services.detector import DetectionService
from app.services.tracker import PersonTracker
from app.services.behavior import BehaviorAnalysisService
from app.services.alert import AlertService
from app.core.config import settings

logger = logging.getLogger(__name__)

def process_video_file(file_path: str, job_id: str, db: Session):
    """
    Process a video file for security analysis
    
    Args:
        file_path: Path to the video file
        job_id: Unique job identifier
        db: Database session
    """
    try:
        # Initialize services
        detector = DetectionService(conf_threshold=settings.DETECTION_CONFIDENCE_THRESHOLD)
        tracker = PersonTracker(iou_threshold=settings.TRACKING_IOU_THRESHOLD)
        
        # Get zone definitions from database
        zone_definitions = [
            {"id": zone.id, "name": zone.name, "coordinates": zone.coordinates}
            for zone in db.query(Zone).all()
        ]
        behavior_analyzer = BehaviorAnalysisService(zone_definitions=zone_definitions)
        alert_service = AlertService()
        
        # Open video file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {file_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Processing video: {file_path}, FPS: {fps}, Duration: {duration}s")
        
        # Process frames
        frame_number = 0
        processing_start = time.time()
        
        # Initialize aggregation data
        daily_stats = {}
        hourly_stats = {}
        zone_stats = {zone["id"]: {"entries": 0, "exits": 0, "total_visitors": 0} for zone in zone_definitions}
        person_zone_history = {}  # track_id -> last_zone_id
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_number += 1
            timestamp = frame_number / fps
            
            # Process every N frames for efficiency
            if frame_number % 5 != 0 and frame_number != 1:
                continue
                
            # Detect persons
            detections = detector.detect_persons(frame)
            
            # Classify persons (staff vs customer, gender)
            detections = detector.classify_persons(frame, detections)
            
            # Track persons
            tracked_persons = tracker.update(detections, frame)
            
            # Analyze behavior
            analytics_data, alerts = behavior_analyzer.analyze_frame(frame, tracked_persons, timestamp)
            
            # Store detection event in database
            detection_event = DetectionEvent(
                timestamp=datetime.now(),
                camera_id="cam_001",  # Example camera ID
                frame_number=frame_number,
                person_count=len(tracked_persons),
                detection_data={
                    "persons": [
                        {
                            "track_id": p.get("track_id"),
                            "bbox": p.get("bbox"),
                            "is_staff": p.get("is_staff", False),
                            "gender": p.get("gender", "unknown"),
                            "zone_id": p.get("zone_id")
                        } for p in tracked_persons
                    ]
                }
            )
            db.add(detection_event)
            
            # Store alerts in database
            for alert_data in alerts:
                alert = Alert(
                    timestamp=datetime.fromisoformat(alert_data["timestamp"]),
                    alert_type=alert_data["alert_type"],
                    severity=alert_data["severity"],
                    zone_id=alert_data["zone_id"],
                    track_id=alert_data["track_id"],
                    description=alert_data["description"]
                )
                db.add(alert)
            
            # Update zone analytics
            for analytics in analytics_data:
                zone_analytics = ZoneAnalytics(
                    zone_id=analytics["zone_id"],
                    timestamp=datetime.fromisoformat(analytics["timestamp"]),
                    time_period="5min",  # 5-minute intervals
                    person_count=analytics["person_count"],
                    avg_dwell_time=analytics["avg_dwell_time"],
                    heat_level=min(10, analytics["person_count"])  # Simple heat level calculation
                )
                db.add(zone_analytics)
            
            # Track zone entries/exits for aggregation
            current_datetime = datetime.now()
            date_key = current_datetime.strftime("%Y-%m-%d")
            hour_key = current_datetime.strftime("%Y-%m-%d %H")
            
            # Initialize date aggregation if needed
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "total_visitors": 0,
                    "peak_visitor_count": 0,
                    "male_count": 0,
                    "female_count": 0,
                    "staff_count": 0,
                    "alert_count": 0
                }
            
            # Initialize hour aggregation if needed
            if hour_key not in hourly_stats:
                hourly_stats[hour_key] = {
                    "total_visitors": 0,
                    "peak_visitor_count": 0,
                    "male_count": 0,
                    "female_count": 0,
                    "staff_count": 0,
                    "alert_count": 0
                }
                
            # Update statistics
            current_person_count = len(tracked_persons)
            daily_stats[date_key]["peak_visitor_count"] = max(
                daily_stats[date_key]["peak_visitor_count"], 
                current_person_count
            )
            hourly_stats[hour_key]["peak_visitor_count"] = max(
                hourly_stats[hour_key]["peak_visitor_count"], 
                current_person_count
            )
            
            # Update gender and staff counts
            for person in tracked_persons:
                track_id = person.get("track_id")
                
                # Count gender
                if person.get("gender") == "male":
                    daily_stats[date_key]["male_count"] += 1
                    hourly_stats[hour_key]["male_count"] += 1
                elif person.get("gender") == "female":
                    daily_stats[date_key]["female_count"] += 1
                    hourly_stats[hour_key]["female_count"] += 1
                
                # Count staff
                if person.get("is_staff", False):
                    daily_stats[date_key]["staff_count"] += 1
                    hourly_stats[hour_key]["staff_count"] += 1
                
                # Track zone transitions
                current_zone = person.get("zone_id")
                if current_zone is not None:
                    if track_id not in person_zone_history:
                        # New person in zone
                        person_zone_history[track_id] = current_zone
                        zone_stats[current_zone]["entries"] += 1
                        zone_stats[current_zone]["total_visitors"] += 1
                        
                        # Count as new visitor
                        daily_stats[date_key]["total_visitors"] += 1
                        hourly_stats[hour_key]["total_visitors"] += 1
                    elif person_zone_history[track_id] != current_zone:
                        # Person moved to a new zone
                        old_zone = person_zone_history[track_id]
                        zone_stats[old_zone]["exits"] += 1
                        zone_stats[current_zone]["entries"] += 1
                        zone_stats[current_zone]["total_visitors"] += 1
                        person_zone_history[track_id] = current_zone
            
            # Count alerts
            daily_stats[date_key]["alert_count"] += len(alerts)
            hourly_stats[hour_key]["alert_count"] += len(alerts)
            
            # Commit to database every 100 frames
            if frame_number % 100 == 0:
                db.commit()
                
                # Log progress
                progress = (frame_number / frame_count) * 100 if frame_count > 0 else 0
                elapsed = time.time() - processing_start
                remaining = (elapsed / frame_number) * (frame_count - frame_number) if frame_number > 0 else 0
                
                logger.info(f"Processing progress: {progress:.1f}%, ETA: {remaining:.1f}s")
        
        # Process is complete, store aggregated statistics
        for date_str, stats in daily_stats.items():
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            for zone_id in zone_stats:
                aggregate = AnalyticsAggregates(
                    date=date_obj,
                    hour=None,  # None for daily aggregates
                    zone_id=zone_id,
                    total_visitors=zone_stats[zone_id]["total_visitors"],
                    peak_visitor_count=stats["peak_visitor_count"],
                    male_count=stats["male_count"],
                    female_count=stats["female_count"],
                    staff_count=stats["staff_count"],
                    alert_count=stats["alert_count"]
                )
                db.add(aggregate)
        
        for hour_str, stats in hourly_stats.items():
            hour_obj = datetime.strptime(hour_str, "%Y-%m-%d %H")
            date_obj = hour_obj.date()
            hour_val = hour_obj.hour
            
            for zone_id in zone_stats:
                aggregate = AnalyticsAggregates(
                    date=date_obj,
                    hour=hour_val,
                    zone_id=zone_id,
                    total_visitors=zone_stats[zone_id]["total_visitors"],
                    peak_visitor_count=stats["peak_visitor_count"],
                    male_count=stats["male_count"],
                    female_count=stats["female_count"],
                    staff_count=stats["staff_count"],
                    alert_count=stats["alert_count"]
                )
                db.add(aggregate)
        
        # Final commit
        db.commit()
        
        # Cleanup
        cap.release()
        if os.path.exists(file_path):
            os.remove(file_path)
        
        processing_time = time.time() - processing_start
        logger.info(f"Video processing complete. Processed {frame_number} frames in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        # Attempt to clean up
        if 'cap' in locals():
            cap.release()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise