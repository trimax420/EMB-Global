import asyncio
import json
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class WebSocketMessage(BaseModel):
    """
    Standardized WebSocket message structure
    """
    type: str
    data: Dict[str, Any] = {}

class ConnectionManager:
    """
    Manages WebSocket connections with advanced features
    """
    def __init__(self):
        # Store active WebSocket connections
        self.active_connections: Set[WebSocket] = set()
        
        # Tracked sessions for specific clients
        self.client_sessions: Dict[str, Set[WebSocket]] = {}
        
        # Message queues for offline/buffering support
        self.message_queues: Dict[str, list] = {}
        
        # Connection tracking
        self._lock = asyncio.Lock()
        
        # Health check and reconnection management
        self.connection_timestamps: Dict[WebSocket, float] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """
        Establish a new WebSocket connection
        
        Args:
            websocket (WebSocket): Incoming WebSocket connection
            client_id (str, optional): Unique identifier for the client
        """
        await websocket.accept()
        
        async with self._lock:
            # Add to global connections
            self.active_connections.add(websocket)
            
            # Track connection timestamp
            self.connection_timestamps[websocket] = asyncio.get_event_loop().time()
            
            # Manage client-specific sessions
            if client_id:
                if client_id not in self.client_sessions:
                    self.client_sessions[client_id] = set()
                self.client_sessions[client_id].add(websocket)
                
                # Send any queued messages
                if client_id in self.message_queues:
                    for message in self.message_queues[client_id]:
                        await self.send_message(websocket, message)
                    # Clear queue after sending
                    del self.message_queues[client_id]
            
            logger.info(f"New WebSocket connection added. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, client_id: str = None):
        """
        Remove a WebSocket connection
        
        Args:
            websocket (WebSocket): Connection to remove
            client_id (str, optional): Client identifier
        """
        try:
            self.active_connections.remove(websocket)
            
            # Remove from client sessions
            if client_id and client_id in self.client_sessions:
                self.client_sessions[client_id].discard(websocket)
                
                # Clean up empty session
                if not self.client_sessions[client_id]:
                    del self.client_sessions[client_id]
            
            # Remove connection timestamp
            del self.connection_timestamps[websocket]
            
            logger.info(f"WebSocket connection removed. Total: {len(self.active_connections)}")
        except KeyError:
            pass

    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Send a message to a specific WebSocket
        
        Args:
            websocket (WebSocket): Target WebSocket
            message (Dict): Message to send
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")

    async def broadcast(self, 
                        message: Dict[str, Any], 
                        filter_type: str = None, 
                        client_id: str = None):
        """
        Broadcast a message to all or filtered connections
        
        Args:
            message (Dict): Message to broadcast
            filter_type (str, optional): Filter connections by message type
            client_id (str, optional): Send to specific client's sessions
        """
        async with self._lock:
            # No active connections
            if not self.active_connections:
                logger.warning("No active WebSocket connections")
                return
            
            # Prepare message
            broadcast_message = {
                "timestamp": asyncio.get_event_loop().time(),
                **message
            }
            
            # Client-specific broadcast
            if client_id:
                if client_id not in self.client_sessions:
                    # Queue message for offline client
                    if client_id not in self.message_queues:
                        self.message_queues[client_id] = []
                    self.message_queues[client_id].append(broadcast_message)
                    return
                
                # Send to specific client's sessions
                target_connections = self.client_sessions[client_id]
            else:
                # Default to all connections
                target_connections = self.active_connections
            
            # Broadcast to filtered connections
            for connection in target_connections:
                try:
                    await self.send_message(connection, broadcast_message)
                except Exception as e:
                    logger.error(f"Broadcast error: {str(e)}")
                    self.disconnect(connection)

    async def clean_stale_connections(self, timeout: float = 300.0):
        """
        Clean up stale WebSocket connections
        
        Args:
            timeout (float): Connection inactivity timeout in seconds
        """
        current_time = asyncio.get_event_loop().time()
        
        async with self._lock:
            stale_connections = [
                ws for ws, timestamp in self.connection_timestamps.items()
                if current_time - timestamp > timeout
            ]
            
            for connection in stale_connections:
                self.disconnect(connection)
                logger.info(f"Removed stale WebSocket connection")

    async def start_cleanup_task(self, interval: float = 300.0):
        """
        Start periodic cleanup of stale connections
        
        Args:
            interval (float): Cleanup interval in seconds
        """
        while True:
            await asyncio.sleep(interval)
            try:
                await self.clean_stale_connections()
            except Exception as e:
                logger.error(f"Connection cleanup error: {str(e)}")

# Global WebSocket connection manager
websocket_manager = ConnectionManager()

# Background task to start cleanup
async def start_websocket_cleanup():
    await websocket_manager.start_cleanup_task()