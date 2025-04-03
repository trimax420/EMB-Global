#!/bin/bash

echo "==================================================="
echo " Security Dashboard - Video File Testing Mode"
echo "==================================================="
echo ""
echo "This script will start the application with video files as camera feeds:"
echo " - Camera 1: cheese_store_NVR_2_NVR_2_20250214160950_20250214161659_844965939.mp4"
echo " - Camera 2: cheese-1.mp4"
echo " - Camera 3: cheese-2.mp4"
echo " - Camera 4: Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4"
echo ""

# Create directory for test frames if it doesn't exist
mkdir -p backend/test_frames

# Set up video files as cameras
export VIDEO_SOURCE_1="cheese_store_NVR_2_NVR_2_20250214160950_20250214161659_844965939.mp4"
export VIDEO_SOURCE_2="cheese-1.mp4"
export VIDEO_SOURCE_3="cheese-2.mp4"
export VIDEO_SOURCE_4="Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4"
export USE_VIDEO_FILES=true
export VIDEO_FILES_DIR="$(pwd)/backend/videos"

# Print settings for debugging
echo "Using VIDEO_FILES_DIR: $VIDEO_FILES_DIR"
echo "Camera 1: $VIDEO_SOURCE_1"
echo "Camera 2: $VIDEO_SOURCE_2"
echo "Camera 3: $VIDEO_SOURCE_3"
echo "Camera 4: $VIDEO_SOURCE_4"

# Start the backend server (FastAPI)
cd backend && python -m uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..
echo "Starting backend server (PID: $BACKEND_PID)..."
sleep 5

# Start the frontend development server
npm run dev &
FRONTEND_PID=$!
echo "Starting frontend server (PID: $FRONTEND_PID)..."
sleep 5

echo ""
echo "==================================================="
echo " Application started! Open http://localhost:5173 in your browser"
echo "==================================================="
echo ""
echo "Press Ctrl+C to shut down the application..."

# Set trap to clean up processes on exit
trap "echo 'Shutting down services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'All services have been stopped.'" EXIT

# Keep script running until user interrupts
wait 