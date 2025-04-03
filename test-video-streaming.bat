@echo off
echo ===================================================
echo  Security Dashboard - Video File Testing Mode
echo ===================================================
echo.
echo This script will start the application with video files as camera feeds:
echo  - Camera 1: cheese_store_NVR_2_NVR_2_20250214160950_20250214161659_844965939.mp4
echo  - Camera 2: cheese-1.mp4
echo  - Camera 3: cheese-2.mp4
echo  - Camera 4: Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4
echo.

REM Create directory for test frames if it doesn't exist
if not exist "backend\test_frames" mkdir backend\test_frames

REM Set up video files as cameras
set VIDEO_SOURCE_1=cheese_store_NVR_2_NVR_2_20250214160950_20250214161659_844965939.mp4
set VIDEO_SOURCE_2=cheese-1.mp4
set VIDEO_SOURCE_3=cheese-2.mp4
set VIDEO_SOURCE_4=Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4
set USE_VIDEO_FILES=true
set VIDEO_FILES_DIR=E:/code/EMB Global/backend/videos

REM Start the backend server (FastAPI)
start cmd /k "cd backend && set VIDEO_FILES_DIR=%VIDEO_FILES_DIR% && set VIDEO_SOURCE_1=%VIDEO_SOURCE_1% && set VIDEO_SOURCE_2=%VIDEO_SOURCE_2% && set VIDEO_SOURCE_3=%VIDEO_SOURCE_3% && set VIDEO_SOURCE_4=%VIDEO_SOURCE_4% && python -m uvicorn main:app --reload --port 8000"
echo Starting backend server...
timeout /t 5 /nobreak

REM Start the frontend development server
start cmd /k "npm run dev"
echo Starting frontend server...
timeout /t 5 /nobreak

echo.
echo ===================================================
echo  Application started! Open http://localhost:5173 in your browser
echo ===================================================
echo.
echo Press any key to shut down the application...
pause > nul

REM Terminate the running processes
taskkill /f /im node.exe > nul 2>&1
taskkill /f /im python.exe > nul 2>&1

echo All services have been stopped.
timeout /t 2 /nobreak 