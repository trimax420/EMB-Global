# Security Dashboard with Real-time Detection

A comprehensive security monitoring dashboard with real-time theft and loitering detection using FastAPI and React.

## Features

- Real-time detection of theft and loitering incidents
- Multiple camera feed monitoring
- Incident history and statistics
- WebSocket-based real-time updates
- Responsive dashboard UI

## Technology Stack

- **Frontend**: React with Material UI
- **Backend**: FastAPI with WebSockets
- **Detection**: Computer vision models for theft and loitering detection
- **Real-time Communication**: WebSockets for live feed and notifications

## Setup Instructions

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- Git

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install frontend dependencies:
   ```
   npm install
   ```

3. Install backend dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

4. Environment Configuration:
   Create a `.env.local` file in the root directory with the following content:
   ```
   # Backend API settings
   VITE_BACKEND_URL=localhost:8000
   VITE_API_BASE_URL=http://localhost:8000/api
   VITE_WS_URL=ws://localhost:8000/api/ws
   
   # Development settings
   VITE_APP_ENV=development
   VITE_ENABLE_MOCK_DATA=true
   ```

### Running the Application

#### Option 1: Start both services together (recommended)

For Windows users:
```
start-all.bat
```

For other platforms:
```
npm run start-all
```

These commands will start both the FastAPI backend and the React frontend.

#### Option 2: Start services separately

Start the FastAPI backend:
```
npm run start-api
```

In a different terminal, start the React frontend:
```
npm run dev
```

### Accessing the Application

- Frontend UI: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Development

- The frontend code is in the `src/` directory
- Backend FastAPI code is in the `backend/` directory
- WebSocket connections are proxied through Vite's development server

## Notes on Real-time Detection

The system uses WebSockets to stream live camera feeds and detection results. The main WebSocket endpoint is `/api/ws/realtime-detection` which accepts the following parameters:

- `detection_type`: Type of detection to perform (`theft`, `loitering`, or `both`)
- `camera_id`: ID of the camera to stream from
- `use_mock`: Set to `true` to use mock data when a real camera is not available
- `use_video`: Set to `true` to use video files from the `/backend/videos` directory

### Using Real Video Files

The system can use video files as camera feeds for testing and demonstration purposes. To use video files:

1. Place your video files in the `/backend/videos` directory
2. The system will automatically use these videos when `use_video=true` is set
3. Each camera ID can be associated with different video files
4. Supported formats include .mp4, .avi, and other formats supported by OpenCV

### Testing Detection

To test the theft and loitering detection using video files, you can use the provided test script:

```bash
cd backend
test_detection.bat
```

This will:
1. Set up the Python environment if needed
2. Run the detection test on a sample video from the `/backend/videos` directory
3. Save frames with detections to the `/backend/test_frames` directory
4. Show a summary of detections found

For more advanced testing, you can run the test script directly:

```bash
python test_detection.py --video your_video.mp4 --frames 200 --detection both
```

## Camera Feed Types

The dashboard supports multiple types of camera feeds:

1. **Real Cameras**: Connected USB or IP cameras
2. **Video Files**: Pre-recorded videos from the videos directory
3. **Mock Feed**: Simulated camera feeds for testing

The UI will indicate the feed type with different colored borders:
- 🟢 Green: Real camera feed
- 🔵 Blue: Video file feed
- 🟠 Orange: Mock data feed
- 🔴 Red: Connection error

## Detection Visualization

The system visualizes detections in the camera feed:

- **Theft Detection**: Red bounding boxes with confidence percentage
- **Loitering Detection**: Blue regions with duration timer

Critical detections are highlighted with alerts and will be recorded as incidents in the system.

To integrate your own detection models, modify the `TheftDetectionModel` and `LoiteringDetectionModel` classes in `backend/app/services/video_processor.py`.
