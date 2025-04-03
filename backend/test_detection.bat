@echo off
echo Testing Detection with Video Files
echo ---------------------------------

REM Check if Python virtual environment exists
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

REM Create test frames directory
if not exist "test_frames" mkdir test_frames

REM Run the detection test with a sample video
set VIDEO_FILE=cheese-1.mp4
if exist "videos\%VIDEO_FILE%" (
    echo Testing detection with %VIDEO_FILE%...
    python test_detection.py --video %VIDEO_FILE% --frames 100 --output test_frames
) else (
    echo Video file not found. Trying another video...
    set VIDEO_FILE=cheese_store_NVR_2_NVR_2_20250214160950_20250214161659_844965939.mp4
    if exist "videos\%VIDEO_FILE%" (
        echo Testing detection with %VIDEO_FILE%...
        python test_detection.py --video %VIDEO_FILE% --frames 100 --output test_frames
    ) else (
        echo No suitable video files found in the videos directory.
        echo Please place video files in the videos directory and try again.
        goto end
    )
)

echo.
echo Test completed!
echo Check the test_frames directory for detection results.
echo.
echo Press any key to view the test frames...
pause > nul

REM Open the test frames directory
start "" "test_frames"

:end
echo.
echo Press any key to exit...
pause > nul 