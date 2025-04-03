#!/usr/bin/env python
"""
WebSocket client tester for the Security Dashboard

Tests the WebSocket server and saves received frames to verify functionality.
"""
import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import os
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory for test frames
OUTPUT_DIR = "test_frames"

async def test_websocket_connection(url, save_frames=True, max_frames=20):
    """
    Test a WebSocket connection by connecting and receiving messages
    
    Args:
        url (str): WebSocket URL to connect to
        save_frames (bool): Whether to save received frames
        max_frames (int): Maximum number of frames to receive before disconnecting
    """
    if save_frames:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    frames_received = 0
    
    try:
        logger.info(f"Connecting to WebSocket at {url}")
        async with websockets.connect(url) as websocket:
            logger.info("Connected!")
            
            while frames_received < max_frames:
                # Receive a message
                message = await websocket.recv()
                
                try:
                    # Parse JSON message
                    data = json.loads(message)
                    message_type = data.get("type", "unknown")
                    
                    logger.info(f"Received message type: {message_type}")
                    
                    if message_type == "connection_established":
                        logger.info(f"Connection established! Server time: {data.get('server_time')}")
                        logger.info(f"Settings: {data.get('settings')}")
                    
                    elif message_type == "inference_result":
                        frames_received += 1
                        logger.info(f"Received frame {frames_received}/{max_frames} - Camera ID: {data.get('camera_id')}")
                        
                        # Check if frame data is present
                        frame_data = data.get("frame")
                        if not frame_data:
                            logger.warning("No frame data received!")
                            continue
                            
                        # Save the frame if requested
                        if save_frames:
                            try:
                                # Decode base64 image
                                img_bytes = base64.b64decode(frame_data)
                                
                                # Convert to numpy array
                                np_arr = np.frombuffer(img_bytes, np.uint8)
                                
                                # Decode to OpenCV image
                                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                
                                if img is None:
                                    logger.error("Failed to decode image!")
                                    continue
                                    
                                # Save image
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                filename = f"{OUTPUT_DIR}/frame_{data.get('camera_id', 'unknown')}_{timestamp}.jpg"
                                cv2.imwrite(filename, img)
                                logger.info(f"Saved frame to {filename}")
                                
                                # Check for detections
                                detections = data.get("detections", {})
                                for det_type, det_data in detections.items():
                                    if det_data.get("detected", False):
                                        logger.info(f"Detection found! Type: {det_type}, Confidence: {det_data.get('confidence', 0)}")
                                
                            except Exception as e:
                                logger.error(f"Error saving frame: {str(e)}")
                    
                    elif message_type == "error":
                        logger.error(f"Error message from server: {data.get('message')}")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
            
            # Send disconnect message
            await websocket.send("disconnect")
            logger.info("Sent disconnect message")
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    
    logger.info(f"Test complete! Received {frames_received} frames.")

def main():
    parser = argparse.ArgumentParser(description="WebSocket Testing Tool")
    parser.add_argument("--url", default="ws://localhost:8000/api/ws/realtime-detection", help="WebSocket URL")
    parser.add_argument("--camera", default="main", help="Camera ID to request")
    parser.add_argument("--detection-type", default="both", choices=["theft", "loitering", "both"], help="Detection type")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to receive")
    parser.add_argument("--no-save", action="store_true", help="Don't save frames")
    
    args = parser.parse_args()
    
    # Build WebSocket URL with query parameters
    ws_url = f"{args.url}?camera_id={args.camera}&detection_type={args.detection_type}"
    
    # Run the test
    asyncio.run(test_websocket_connection(ws_url, not args.no_save, args.frames))

if __name__ == "__main__":
    main() 