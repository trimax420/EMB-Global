@echo off
echo Testing WebSocket video streaming...

REM Set up video files environment variables
set VIDEO_FILES_DIR=E:\code\EMB Global\backend\videos
set USE_VIDEO_FILES=true

REM Set specific video source for each camera ID
set VIDEO_SOURCE_1=cheese_store_NVR_2_NVR_2_20250214160950_20250214161659_844965939.mp4
set VIDEO_SOURCE_2=cheese-1.mp4
set VIDEO_SOURCE_3=cheese-2.mp4
set VIDEO_SOURCE_4=Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4

REM Change to the backend directory and start the server
cd backend
echo Using VIDEO_FILES_DIR: %VIDEO_FILES_DIR%
echo Camera 1: %VIDEO_SOURCE_1%
echo Camera 2: %VIDEO_SOURCE_2%
echo Camera 3: %VIDEO_SOURCE_3%
echo Camera 4: %VIDEO_SOURCE_4%
python -m uvicorn main:app --reload --port 8000 