from fastapi import APIRouter, HTTPException, WebSocket, Request, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import asyncio
import json
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaRelay, MediaBlackhole
import av
import uuid
import cv2
import numpy as np
from datetime import datetime
from app.core.config import settings
from app.core.video_processor import process_frame, initialize_model, get_model
from app.core.theft_detector import process_frame_for_theft

logger = logging.getLogger(__name__)
router = APIRouter()

# Store active peer connections
active_connections: Dict[str, RTCPeerConnection] = {}
# Store active camera subscriptions
camera_subscriptions: Dict[str, List[str]] = {}

# Media relay for forwarding video
relay = MediaRelay()

class SignalingMessage(BaseModel):
    type: str
    sdp: Optional[dict] = None
    candidate: Optional[dict] = None
    camera_id: Optional[str] = None

@router.post("/signal")
async def webrtc_signal(message: SignalingMessage, background_tasks: BackgroundTasks):
    """
    Handle WebRTC signaling messages (offer, answer, ice candidate)
    """
    logger.info(f"Received signaling message: {message.type}")
    
    if message.type == "offer":
        # Create a new RTCPeerConnection
        peer_connection = RTCPeerConnection()
        connection_id = str(uuid.uuid4())
        active_connections[connection_id] = peer_connection
        
        # Set up a data channel for control messages
        @peer_connection.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel established: {channel.label}")
            
            @channel.on("message")
            async def on_message(message_str):
                try:
                    # Parse the message
                    message = json.loads(message_str)
                    action = message.get("action")
                    
                    if action == "subscribe":
                        # Subscribe to camera
                        camera_id = message.get("camera_id")
                        if camera_id:
                            if camera_id not in camera_subscriptions:
                                camera_subscriptions[camera_id] = []
                            
                            camera_subscriptions[camera_id].append(connection_id)
                            
                            # Send confirmation
                            await channel.send(json.dumps({
                                "type": "subscribed",
                                "camera_id": camera_id
                            }))
                            
                            logger.info(f"Client {connection_id} subscribed to camera {camera_id}")
                    
                    elif action == "unsubscribe":
                        # Unsubscribe from camera
                        camera_id = message.get("camera_id")
                        if camera_id and camera_id in camera_subscriptions:
                            if connection_id in camera_subscriptions[camera_id]:
                                camera_subscriptions[camera_id].remove(connection_id)
                            
                            await channel.send(json.dumps({
                                "type": "unsubscribed",
                                "camera_id": camera_id
                            }))
                            
                            logger.info(f"Client {connection_id} unsubscribed from camera {camera_id}")
                    
                    elif action == "start_inference":
                        # Start inference on camera
                        camera_id = message.get("camera_id")
                        detection_types = message.get("detection_types", ["object"])
                        
                        # Set up video frame handling
                        # This would connect to the camera and start processing frames
                        background_tasks.add_task(
                            start_camera_stream, 
                            connection_id=connection_id, 
                            camera_id=camera_id,
                            peer_connection=peer_connection,
                            detection_types=detection_types
                        )
                        
                        await channel.send(json.dumps({
                            "type": "inference_started",
                            "camera_id": camera_id
                        }))
                        
                        logger.info(f"Started inference for camera {camera_id}")
                    
                    elif action == "stop_inference":
                        # Stop inference
                        camera_id = message.get("camera_id")
                        
                        # Code to stop the video stream would go here
                        # For now, just acknowledge
                        await channel.send(json.dumps({
                            "type": "inference_stopped",
                            "camera_id": camera_id
                        }))
                        
                        logger.info(f"Stopped inference for camera {camera_id}")
                    
                    else:
                        logger.warning(f"Unknown action: {action}")
                        await channel.send(json.dumps({
                            "type": "error",
                            "message": f"Unknown action: {action}"
                        }))
                
                except Exception as e:
                    logger.error(f"Error processing data channel message: {str(e)}")
                    await channel.send(json.dumps({
                        "type": "error",
                        "message": f"Error: {str(e)}"
                    }))
        
        # Set up connection state change handler
        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state changed to: {peer_connection.connectionState}")
            if peer_connection.connectionState == "failed" or peer_connection.connectionState == "closed":
                # Clean up the connection
                if connection_id in active_connections:
                    del active_connections[connection_id]
                
                # Remove from camera subscriptions
                for camera_id, connections in camera_subscriptions.items():
                    if connection_id in connections:
                        connections.remove(connection_id)
        
        # Set the remote description from the offer
        offer = RTCSessionDescription(sdp=message.sdp["sdp"], type=message.sdp["type"])
        await peer_connection.setRemoteDescription(offer)
        
        # Create an answer
        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)
        
        # Return the answer
        return {
            "type": "answer",
            "sdp": {"sdp": peer_connection.localDescription.sdp, "type": peer_connection.localDescription.type}
        }
    
    elif message.type == "ice_candidate" and message.candidate:
        # Handle ICE candidate
        # This would require storing the peer connection by some ID
        # For now, just log it
        logger.info(f"Received ICE candidate: {message.candidate}")
        return {"status": "acknowledged"}
    
    else:
        # Unknown message type
        raise HTTPException(status_code=400, detail=f"Unknown message type: {message.type}")

async def start_camera_stream(connection_id: str, camera_id: str, peer_connection: RTCPeerConnection, detection_types: List[str]):
    """
    Start streaming video from camera and process frames
    """
    try:
        # This is a placeholder for actual camera connection
        # In a real implementation, you would connect to a camera by ID
        
        # For demo purposes, we'll create a video track with black frames
        # that will be processed with detections
        
        class DetectionVideoStreamTrack(aiortc.MediaStreamTrack):
            kind = "video"
            
            def __init__(self):
                super().__init__()
                self.frame_count = 0
                self.width = 640
                self.height = 480
                
                # Initialize models based on detection types
                self.models = {}
                if "object" in detection_types:
                    self.models["object"] = get_model("object")
                if "theft" in detection_types:
                    self.models["theft"] = get_model("theft")
                if "loitering" in detection_types:
                    self.models["loitering"] = get_model("loitering")
            
            async def recv(self):
                # This would get frames from the actual camera
                # For now, generate a test frame
                
                # Create a test frame (black with timestamp)
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
                # Add some visual elements (grid lines)
                for i in range(0, self.width, 40):
                    cv2.line(frame, (i, 0), (i, self.height), (30, 30, 50), 1)
                for i in range(0, self.height, 40):
                    cv2.line(frame, (0, i), (self.width, i), (30, 30, 50), 1)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"{timestamp} - Camera {camera_id}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {self.frame_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                # Generate some test detections
                detections = []
                
                # Occasionally add a "person" detection
                if self.frame_count % 30 < 20:
                    x = 100 + (self.frame_count % 100)
                    y = 100 + (self.frame_count % 50)
                    detections.append({
                        "type": "object",
                        "class_name": "person",
                        "confidence": 0.85,
                        "bbox": [x, y, x + 100, y + 200]
                    })
                    
                    # Draw the detection
                    cv2.rectangle(frame, (x, y), (x + 100, y + 200), (50, 100, 200), 2)
                    cv2.putText(frame, "person 85%", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 200), 1)
                
                # Occasionally add a "theft" detection
                if self.frame_count % 100 > 80:
                    x = 300
                    y = 150
                    detections.append({
                        "type": "theft",
                        "confidence": 0.75,
                        "bbox": [x, y, x + 120, y + 180],
                        "zone": "chest"
                    })
                    
                    # Draw the detection with red color
                    cv2.rectangle(frame, (x, y), (x + 120, y + 180), (0, 0, 200), 2)
                    cv2.putText(frame, "theft 75%", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
                
                # Send detection data through the data channel
                if peer_connection.dataChannels:
                    for channel in peer_connection.dataChannels.values():
                        if channel.readyState == "open":
                            try:
                                await channel.send(json.dumps({
                                    "type": "detections",
                                    "camera_id": camera_id,
                                    "detections": detections,
                                    "frame_number": self.frame_count,
                                    "timestamp": datetime.now().isoformat()
                                }))
                            except Exception as e:
                                logger.error(f"Error sending detection data: {str(e)}")
                
                # Increment frame counter
                self.frame_count += 1
                
                # Convert to video frame
                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                video_frame.pts = self.frame_count * 1000  # millisecond timestamp
                video_frame.time_base = av.Fraction(1, 1000)  # millisecond timebase
                
                # Simulate processing time
                await asyncio.sleep(0.1)
                
                return video_frame
        
        # Create a video stream track
        video_track = DetectionVideoStreamTrack()
        
        # Add the track to the peer connection
        peer_connection.addTrack(relay.subscribe(video_track))
        
        # The track will run until the connection is closed
        # We need to keep a reference to prevent garbage collection
        peer_connection._video_track = video_track
        
        logger.info(f"Started camera stream for {camera_id}")
        
    except Exception as e:
        logger.error(f"Error starting camera stream: {str(e)}")
        # Try to send error message via data channel
        if peer_connection.dataChannels:
            for channel in peer_connection.dataChannels.values():
                if channel.readyState == "open":
                    try:
                        await channel.send(json.dumps({
                            "type": "error",
                            "message": f"Camera stream error: {str(e)}"
                        }))
                    except:
                        pass 