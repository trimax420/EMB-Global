from fastapi import WebSocket, status
from typing import Dict, List
import logging
import json
import traceback

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manager for WebSocket connections supporting multiple clients
    """
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """
        Accept a WebSocket connection and store it
        """
        try:
            # Generate client ID if not provided
            if client_id is None:
                import uuid
                client_id = f"client_{uuid.uuid4().hex[:8]}"
                
            # Accept the connection
            await websocket.accept()
            logger.info(f"WebSocket connection accepted: {client_id}")
            
            # Store the connection
            self.active_connections[client_id] = websocket
            
            # Send initial connection confirmation
            await websocket.send_json({
                "type": "connection_established",
                "client_id": client_id
            })
            
            return client_id
            
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {str(e)}")
            logger.error(traceback.format_exc())
            # Try to close with error if possible
            try:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except Exception as close_error:
                logger.error(f"Error closing WebSocket: {str(close_error)}")
            raise
    
    def disconnect(self, websocket: WebSocket, client_id: str = None):
        """
        Remove a WebSocket connection
        """
        try:
            if client_id and client_id in self.active_connections:
                logger.info(f"Client disconnected: {client_id}")
                del self.active_connections[client_id]
            else:
                # Find by websocket object
                for cid, ws in list(self.active_connections.items()):
                    if ws == websocket:
                        logger.info(f"Client disconnected: {cid}")
                        del self.active_connections[cid]
                        break
        except Exception as e:
            logger.error(f"Error disconnecting client: {str(e)}")
    
    async def broadcast(self, message):
        """
        Broadcast message to all connected clients
        """
        if not self.active_connections:
            logger.debug("No active connections for broadcast")
            return
            
        # Convert message to JSON if it's a dict
        if isinstance(message, dict):
            message_str = json.dumps(message)
        else:
            message_str = message
            
        # Send to all active connections
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                if isinstance(message, dict):
                    await websocket.send_json(message)
                else:
                    await websocket.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(None, client_id)
            
    def get_active_connections(self):
        """Get number of active connections"""
        return len(self.active_connections)

# Global WebSocket manager instance
websocket_manager = ConnectionManager()