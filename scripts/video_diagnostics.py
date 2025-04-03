#!/usr/bin/env python3
"""
Video File Diagnostics Tool

This script tests if OpenCV can open and read video files with different path formats.
It helps identify issues with file paths, especially on Windows systems.
"""

import os
import sys
import cv2
import argparse
import glob
from pathlib import Path

def check_video_file(video_path):
    """Check if a video file can be opened by OpenCV and get its properties."""
    print(f"\nChecking video file: {video_path}")
    
    # Basic path info
    print(f"  Exists: {os.path.exists(video_path)}")
    print(f"  Absolute path: {os.path.abspath(video_path)}")
    print(f"  File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB" if os.path.exists(video_path) else "N/A")
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  [ERROR] OpenCV could not open the file")
        
        # Try with different slash direction
        if '/' in video_path:
            alt_path = video_path.replace('/', '\\')
            print(f"  Trying alternative path: {alt_path}")
            cap = cv2.VideoCapture(alt_path)
            if cap.isOpened():
                print("  [SUCCESS] OpenCV opened the file with backslashes")
            else:
                print("  [ERROR] OpenCV still could not open the file with backslashes")
        elif '\\' in video_path:
            alt_path = video_path.replace('\\', '/')
            print(f"  Trying alternative path: {alt_path}")
            cap = cv2.VideoCapture(alt_path)
            if cap.isOpened():
                print("  [SUCCESS] OpenCV opened the file with forward slashes")
            else:
                print("  [ERROR] OpenCV still could not open the file with forward slashes")
    else:
        print("  [SUCCESS] OpenCV opened the file successfully")
        
    # Get video properties if opened
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Video properties:")
        print(f"    Resolution: {width}x{height}")
        print(f"    FPS: {fps}")
        print(f"    Frame count: {frame_count}")
        
        # Try to read first frame
        ret, frame = cap.read()
        if ret:
            print(f"    First frame read successfully: {frame.shape}")
        else:
            print(f"    [ERROR] Could not read first frame")
        
        # Try to read a later frame
        if frame_count > 100:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
            ret, frame = cap.read()
            if ret:
                print(f"    Frame #100 read successfully")
            else:
                print(f"    [ERROR] Could not read frame #100")
        
        # Release the capture
        cap.release()

def auto_discover_videos(base_dir=None):
    """Automatically discover video files in directories."""
    if base_dir is None:
        base_dir = os.getcwd()
        
    print(f"Searching for video files in: {base_dir}")
    
    # Common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    # Look for upload directories
    upload_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'upload' in dir_name.lower() or 'raw' in dir_name.lower():
                upload_dirs.append(os.path.join(root, dir_name))
    
    # Look for video files in upload directories first
    for upload_dir in upload_dirs:
        print(f"Checking upload directory: {upload_dir}")
        for ext in video_extensions:
            pattern = os.path.join(upload_dir, f"*{ext}")
            files = glob.glob(pattern)
            video_files.extend(files)
    
    # If no videos found in upload dirs, search more broadly
    if not video_files:
        for ext in video_extensions:
            pattern = os.path.join(base_dir, f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            video_files.extend(files)
    
    # Limit the number of files
    video_files = video_files[:10]  # Only check the first 10 videos to avoid overwhelm
    
    print(f"Found {len(video_files)} video files")
    return video_files

def main():
    parser = argparse.ArgumentParser(description='Video File Diagnostics Tool')
    parser.add_argument('--video', help='Path to the video file to check')
    parser.add_argument('--discover', action='store_true', help='Auto-discover video files')
    args = parser.parse_args()
    
    print(f"Running on: {sys.platform}")
    print(f"Current working directory: {os.getcwd()}")
    
    if args.discover:
        video_files = auto_discover_videos()
        for video_file in video_files:
            check_video_file(video_file)
    elif args.video:
        check_video_file(args.video)
    else:
        # Test a few windows-style paths for the known video files
        test_paths = [
            "E:\\code\\EMB Global\\backend\\uploads\\raw\\cheese-1.mp4",
            "E:/code/EMB Global/backend/uploads/raw/cheese-1.mp4",
            "E:\\code\\EMB Global\\backend\\uploads\\raw\\cheese-2.mp4",
            "E:/code/EMB Global/backend/uploads/raw/cheese-2.mp4",
            "E:\\code\\EMB Global\\backend\\uploads\\raw\\Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4",
            "E:/code/EMB Global/backend/uploads/raw/Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4",
        ]
        
        for path in test_paths:
            check_video_file(path)

if __name__ == "__main__":
    main() 