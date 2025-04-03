#!/usr/bin/env python
"""
Test script for theft and loitering detection using video files
This script helps verify that our detection logic is working correctly
"""

import cv2
import os
import time
import argparse
import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).parent))

from app.services.video_processor import VideoProcessor, VideoFileCamera
from app.services.detection_models import TheftDetectionModel, LoiteringDetectionModel

async def process_frames(args):
    """Process frames asynchronously using the new async method signature"""
    # Initialize video processor and cameras
    video_processor = VideoProcessor()
    
    # Initialize detection models
    theft_model = None
    loitering_model = None
    
    if args.detection in ["theft", "both"]:
        print("Initializing theft detection model...")
        theft_model = TheftDetectionModel()
    
    if args.detection in ["loitering", "both"]:
        print("Initializing loitering detection model...")
        loitering_model = LoiteringDetectionModel()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process frames and run detection
    total_frames = 0
    theft_detected_count = 0
    loitering_detected_count = 0
    
    start_time = time.time()
    last_fps_update = start_time
    fps = 0
    
    print(f"Processing up to {args.frames} frames...")
    
    # Set up video file if specified
    if args.video:
        video_path = os.path.join("videos", args.video)
        print(f"Using video file: {video_path}")
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
            
        video_camera = VideoFileCamera(args.camera_id, specific_video=video_path)
        video_processor.video_cameras[args.camera_id] = video_camera
    
    for frame_idx in range(args.frames):
        # Get frame
        frame, is_mock, video_info, is_video_file = await video_processor.get_video_frame(
            camera_id=args.camera_id, use_mock=True, use_video=True
        )
        
        if frame is None:
            print(f"No more frames available after {total_frames} frames")
            break
        
        # Update counters
        total_frames += 1
        current_time = time.time()
        
        # Update FPS every second
        if current_time - last_fps_update >= 1.0:
            elapsed = current_time - start_time
            fps = total_frames / elapsed
            print(f"Processing at {fps:.1f} FPS - Frame {total_frames}/{args.frames}")
            last_fps_update = current_time
        
        # Run theft detection if enabled
        if theft_model:
            theft_results = theft_model.detect(frame.copy())
            
            if theft_results["detected"]:
                theft_detected_count += 1
                confidence = theft_results["confidence"]
                print(f"Frame {total_frames}: Theft detected with {confidence:.2f} confidence")
                
                # Save frame with detections
                theft_frame = frame.copy()
                for box in theft_results["bounding_boxes"]:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(theft_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(theft_frame, f"Theft: {confidence:.2f}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Save the frame
                cv2.imwrite(f"{args.output}/theft_detection_{total_frames:04d}.jpg", theft_frame)
        
        # Run loitering detection if enabled
        if loitering_model:
            loitering_results = loitering_model.detect(frame.copy(), camera_id=args.camera_id)
            
            if loitering_results["detected"]:
                loitering_detected_count += 1
                duration = loitering_results["duration"]
                print(f"Frame {total_frames}: Loitering detected with {duration:.2f}s duration")
                
                # Save frame with detections
                loitering_frame = frame.copy()
                for region in loitering_results["regions"]:
                    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                    cv2.rectangle(loitering_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(loitering_frame, f"Loitering: {duration:.2f}s", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Save the frame
                cv2.imwrite(f"{args.output}/loitering_detection_{total_frames:04d}.jpg", loitering_frame)
        
        # Show progress every 10 frames
        if total_frames % 10 == 0:
            print(f"Processed {total_frames} frames - Theft detections: {theft_detected_count}, Loitering detections: {loitering_detected_count}")
    
    # Print summary
    elapsed_time = time.time() - start_time
    avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
    
    print("\n--- Detection Test Summary ---")
    print(f"Total frames processed: {total_frames}")
    print(f"Average processing speed: {avg_fps:.1f} FPS")
    print(f"Theft detections: {theft_detected_count} ({theft_detected_count/total_frames*100:.1f}%)")
    print(f"Loitering detections: {loitering_detected_count} ({loitering_detected_count/total_frames*100:.1f}%)")
    print(f"Output frames saved to: {args.output}/")
    print("----------------------------")
    
    # Cleanup
    video_processor.release_video_cameras(args.camera_id)
    print("Test complete!")

def main():
    parser = argparse.ArgumentParser(description='Test theft and loitering detection with video files')
    parser.add_argument('--video', type=str, help='Path to video file to use (relative to the videos directory)')
    parser.add_argument('--camera_id', type=str, default="1", help='Camera ID to use')
    parser.add_argument('--detection', type=str, default="both", choices=['theft', 'loitering', 'both'], help='Detection type to test')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to process')
    parser.add_argument('--output', type=str, default="test_frames", help='Output directory for frames with detections')
    
    args = parser.parse_args()
    
    # Run the async processing function
    asyncio.run(process_frames(args))

if __name__ == "__main__":
    main() 