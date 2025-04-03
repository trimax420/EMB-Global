#!/usr/bin/env python3
"""
WebSocket test client for real-time detection streaming.
Run this script to test the WebSocket functionality.
"""

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import argparse

async def test_detection_streaming(server_url="ws://localhost:8000/api/ws/realtime-detection", 
                                   detection_type="both", 
                                   camera_id="0"):
    """Test the real-time detection streaming endpoint"""
    
    connection_url = f"{server_url}?detection_type={detection_type}&camera_id={camera_id}"
    print(f"Connecting to {connection_url}")
    
    try:
        async with websockets.connect(connection_url) as websocket:
            print(f"Connected to WebSocket server")
            
            # Receive and process messages
            while True:
                try:
                    # Wait for message
                    message = await websocket.recv()
                    
                    # Parse JSON message
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "connection_established":
                        print(f"Connection established: {data}")
                    
                    elif msg_type == "inference_result":
                        timestamp = data.get("timestamp")
                        camera_id = data.get("camera_id")
                        detections = data.get("detections", {})
                        
                        print(f"Received inference at {timestamp} for camera {camera_id}")
                        print(f"Detections: {detections}")
                        
                        # Display frame if available
                        if "frame" in data:
                            try:
                                frame_data = base64.b64decode(data["frame"])
                                frame_np = np.frombuffer(frame_data, np.uint8)
                                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                                
                                # Draw detection boxes if present
                                if "theft" in detections and detections["theft"].get("detected"):
                                    for bbox in detections["theft"].get("bounding_boxes", []):
                                        x1, y1, x2, y2 = bbox
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        cv2.putText(frame, "Theft", (x1, y1-10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
                                if "loitering" in detections and detections["loitering"].get("detected"):
                                    for region in detections["loitering"].get("regions", []):
                                        x, y, w, h = region.get("x", 0), region.get("y", 0), \
                                                     region.get("width", 0), region.get("height", 0)
                                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                        cv2.putText(frame, "Loitering", (x, y-10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                
                                # Display frame
                                cv2.imshow("Real-time Detection", frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            except Exception as e:
                                print(f"Error displaying frame: {str(e)}")
                    
                    elif msg_type == "error":
                        print(f"Error from server: {data.get('message')}")
                    
                    elif msg_type == "ping":
                        # Send pong response
                        await websocket.send(json.dumps({"type": "pong"}))
                    
                    else:
                        print(f"Unknown message type: {msg_type}")
                        print(f"Data: {data}")
                
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break
    
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test WebSocket real-time detection")
    parser.add_argument("--url", default="ws://localhost:8000/api/ws/realtime-detection", 
                        help="WebSocket server URL")
    parser.add_argument("--type", default="both", choices=["theft", "loitering", "both"],
                        help="Detection type")
    parser.add_argument("--camera", default="0", help="Camera ID")
    
    args = parser.parse_args()
    
    asyncio.run(test_detection_streaming(args.url, args.type, args.camera))
