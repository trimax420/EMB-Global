# WebRTC Implementation for Real-Time Video Detection

This project implements WebRTC for real-time video streaming and object detection, providing a more efficient alternative to WebSockets for low-latency video transmission.

## Features

- Direct peer-to-peer video streaming from server to browser
- Low-latency video transmission (compared to WebSockets)
- Real-time object detection annotations overlaid on video
- Automatic fallback to WebSocket or mock data when WebRTC is not available
- Data channel for control messages and detection results

## Requirements

### Backend Dependencies
- aiortc (WebRTC implementation for Python)
- av (for video frame processing)
- opencv-python (for image processing)
- numpy (for numerical operations)

### Frontend Dependencies
- Modern browser with WebRTC support (Chrome, Firefox, Safari, Edge)

## Setup

1. Install the required backend dependencies:
   ```
   pip install -r requirements.txt
   ```

2. The frontend component automatically uses WebRTC when available.

## How It Works

### Frontend
The `RealTimeDetection.jsx` component:
1. Attempts to establish a WebRTC connection first
2. Falls back to WebSocket if WebRTC fails
3. Uses mock data as a last resort when both are unavailable
4. Displays video stream from WebRTC or renders frames from WebSocket/mock data on canvas

### Backend
The WebRTC implementation:
1. Handles signaling for WebRTC connection setup
2. Creates a video track with processed frames from cameras
3. Sends detection data via the data channel
4. Manages client subscriptions to different cameras

## API Endpoints

- `POST /api/webrtc/signal`: WebRTC signaling endpoint that handles:
  - Offers from clients
  - ICE candidates
  - Connection management

## Usage

In the frontend component, WebRTC is enabled by default. The component will attempt to:

1. Connect via WebRTC
2. Process video frames with real-time detection
3. Display the results with low latency

The component will automatically fall back to WebSocket or mock detection if any step fails.

## Troubleshooting

- If WebRTC connection fails, check browser console for detailed error messages
- Ensure the backend server has proper access to cameras
- For local development, make sure to use HTTPS or localhost (required for WebRTC)
- Check that all required dependencies are installed on the server

## Security Considerations

- WebRTC connections are encrypted by default
- For production, configure proper STUN/TURN servers for NAT traversal
- Implement authentication for the signaling server

## Future Improvements

- Add STUN/TURN server configuration for NAT traversal
- Implement reconnection logic for lost connections
- Add support for audio if needed
- Optimize frame processing for better performance 