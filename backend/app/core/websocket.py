import logging
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    WebSocket connection manager for handling real-time communication
    with clients.
    """
    
    def __init__(self):
        # Active connections: Map of connection ID -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """
        Accept a WebSocket connection and store it for future use.
        
        Args:
            websocket: The WebSocket connection to store
            client_id: Optional client identifier. If not provided, the websocket ID is used.
        """
        await websocket.accept()
        
        # Generate client ID if not provided
        connection_id = client_id or str(id(websocket))
        self.active_connections[connection_id] = websocket
        logger.info(f"Client connected: {connection_id}")
        
        return connection_id
        
    def disconnect(self, client_id: str):
        """
        Remove a WebSocket connection.
        
        Args:
            client_id: The ID of the client to remove
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client disconnected: {client_id}")
            
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """
        Send a message to a specific client.
        
        Args:
            message: The message to send (will be converted to JSON)
            client_id: The ID of the client to send the message to
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)
            
    async def broadcast(self, message: Dict[str, Any], exclude: List[str] = None):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: The message to send (will be converted to JSON)
            exclude: List of client IDs to exclude from the broadcast
        """
        if exclude is None:
            exclude = []
            
        for client_id, websocket in list(self.active_connections.items()):
            if client_id not in exclude:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to client {client_id}: {str(e)}")
                    # Remove the connection if it's broken
                    self.disconnect(client_id)
                    
    def get_connection_count(self) -> int:
        """
        Get the number of active connections.
        
        Returns:
            int: Number of active connections
        """
        return len(self.active_connections) 